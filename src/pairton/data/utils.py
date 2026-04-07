from itertools import product
from collections.abc import Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import cast, TYPE_CHECKING

import jaxtyping as jx
import torch as th
from torch.utils.data import default_collate

from pairton.data.types import (
    SequenceBatch,
    SequenceBatches,
)
from pairton.data.transforms import BaseTransform
from ml_utils.utils import exists

def construct_adjacency(
    features: jx.Float[th.Tensor, "num_jets num_features"],
    groups: list[jx.Int[th.Tensor, " num_jets"]],
) -> jx.Int[th.Tensor, "num_jets num_jets"]:
    adjacency = th.eye(len(features), dtype=th.int64)
    for group in groups:
        for idx1, idx2 in product(group, repeat=2):
            if idx1 < 0 or idx2 < 0:  # -1 is used as a placeholder for no association
                continue
            adjacency[idx1, idx2] ^= 1  # Sets off-diagonal to 1 and diagonal to 0.
    return adjacency


def _verify_group(
    group: jx.Int[th.Tensor, " 3"],
    max_index: int,
):
    for idx in group:
        if idx < 0:
            raise ValueError("Group contains invalid index.")
        if idx >= max_index:
            raise ValueError("Group index out of bounds.")


def _mask_row_and_column(
    matrix: jx.Bool[th.Tensor, "num_jets num_jets"],
    index: int,
) -> jx.Bool[th.Tensor, "num_jets num_jets"]:
    matrix[index, :] = True
    matrix[:, index] = True
    return matrix


def _mask_all_intersections(
    matrix: jx.Bool[th.Tensor, "num_jets num_jets"],
    indices: list[int],
) -> jx.Bool[th.Tensor, "num_jets num_jets"]:
    for idx1, idx2 in product(indices, repeat=2):
        matrix[idx1, idx2] = True
    return matrix


def _mask_group_row_and_column(
    matrix: jx.Bool[th.Tensor, "num_jets num_jets"],
    group: jx.Int[th.Tensor, " num_jets"],
) -> jx.Bool[th.Tensor, "num_jets num_jets"]:
    for idx in group:
        matrix = _mask_row_and_column(matrix, int(idx))
    return matrix


def mask_adjacency(
    num_jets: int,
    mask_level: int,
    groups: list[jx.Int[th.Tensor, " 3"]],
) -> jx.Int[th.Tensor, "num_jets num_jets 2"]:
    """Creates a visibility mask for discrete diffusion adjacency prediction.

    True values indicate visible (unmasked) entries, False values indicate masked entries.

    Args:
        num_jets: Number of jets in the event.
        mask_level: Level of masking to apply.
        groups: List of jet groups for masking.

    Returns:
        A boolean tensor of shape (num_jets, num_jets, 2) indicating visibility.
            First channel corresponds to top adjacency, second to w adjacency.

    """
    for group in groups:
        _verify_group(group, num_jets)
    for group1_idx, group2_idx in product(groups[0], groups[1]):
        if group1_idx == group2_idx and group1_idx >= 0:
            raise ValueError("Groups must be disjoint.")

    # True values are visible, False values are masked.
    adjacency_mask = th.zeros((num_jets, num_jets, 2), dtype=th.bool)
    if mask_level == 0:
        return adjacency_mask
    if mask_level == 1:  # One W visible
        group_idx = th.rand(1).item() > 0.5
        group = groups[int(group_idx)]
        w1, w2 = group[1:]
        if w1 < 0 or w2 < 0:
            raise ValueError("Need fully reconstructable jets for mask levels.")

        adjacency_mask[..., 1] = _mask_row_and_column(adjacency_mask[..., 1], int(w1))
        adjacency_mask[..., 1] = _mask_row_and_column(adjacency_mask[..., 1], int(w2))
        adjacency_mask[..., 0] = _mask_all_intersections(
            adjacency_mask[..., 0], [int(w1), int(w2)]
        )

    elif mask_level == 2:  # Both W's visible
        adjacency_mask[..., 1] = True
        w1, w2 = groups[0][1:]
        w3, w4 = groups[1][1:]
        all_ws = [int(w1), int(w2), int(w3), int(w4)]
        adjacency_mask[..., 0] = _mask_all_intersections(adjacency_mask[..., 0], all_ws)

    elif mask_level == 3:  # Both W's and 1 top
        group_idx = th.rand(1).item() > 0.5
        fully_visible_group = groups[int(group_idx)]
        other_group = groups[1 - int(group_idx)]
        adjacency_mask[..., 1] = True
        adjacency_mask[..., 0] = _mask_group_row_and_column(
            adjacency_mask[..., 0], fully_visible_group
        )
        adjacency_mask[..., 0] = _mask_all_intersections(
            adjacency_mask[..., 0], other_group[1:].tolist()
        )

    else:
        raise ValueError("Invalid mask level.")

    return adjacency_mask


@dataclass
class SpanetH5Structure:
    """Structure of the HDF5 file for TTBar datasets.

    Attributes:
        feature_key:
            Key for the group containing the input features.
        mask_key:
            Key for the dataset containing the input mask.
        groups:
            Sequence of keys for the target groups in the dataset.
    """

    feature_key: str = "INPUTS/Source"
    mask_key: str = "INPUTS/Source/MASK"
    groups: Sequence[str] = ("TARGETS/t1", "TARGETS/t2")


@dataclass
class SpanetDatasetConfig:
    """Configuration for TTBar datasets.

    Attributes:
        features:
            Sequence of transformations to apply to the dataset features.
        _structure:
            Structure of the HDF5 file, initialized to None by default.
        num_events:
            Number of events in the dataset, can be set to None if not known.

    """

    features: Sequence[BaseTransform]
    structure: SpanetH5Structure = field(default_factory=SpanetH5Structure)
    num_events: int | None = None


@dataclass
class TopographH5Structure:
    """Structure of the HDF5 file for Topograph datasets.

    Attributes:
        feature_key:
            Key for the group containing the jet features.
        matchability_key:
            Key for the dataset containing the matchability criteria.
        jet_indices_key:
            Key for the dataset containing the jet indices.
        num_bjets_key:
            Key for the dataset containing the number of b-jets.
        num_jets_key:
            Key for the dataset containing the number of jets.
    """

    feature_key: str = "delphes/jets"
    matchability_key: str = "delphes/matchability"
    jet_indices_key: str = "delphes/jets_indices"
    num_bjets_key: str = "delphes/nbjets"
    num_jets_key: str = "delphes/njets"


@dataclass
class TopographDatasetConfig:
    """Configuration for Topograph datasets.

    Attributes:
        features:
            Sequence of transformations to apply to the dataset features.
        matchability_criteria:
            Matchability criteria to filter events, can be an integer or a list of
            integers. The integers are treated as bitmasks and if the bitmask of a
            given event contains any of the bitmasks in the list, the event is kept.
            See https://zenodo.org/records/7737248 for more details.
        structure:
            Structure of the HDF5 file, initialized to None by default.
            see `TopographH5Structure` for details.
        num_jet_range:
            Tuple specifying the (min, max) number of required jets. Both ends are
            inclusive.
        num_bjet_range:
            Tuple specifying the (min, max) number of required b-jets. Both ends are
            inclusive.
        max_num_events:
            Maximum number of events to load from the dataset, can be set to None to
            load all events.
        random_seed:
            Random seed for shuffling the dataset before applying the max_num_events
        group_indices:
            List of lists specifying the indices of jets to form each group.
            Defaults to [[0, 1, 2], [3, 4, 5]] which gives two groups, one for each
            top hadron.
    """

    features: Sequence[BaseTransform]
    matchability_criteria: int | list[int] = 0b000000
    structure: TopographH5Structure = field(default_factory=TopographH5Structure)
    num_jet_range: tuple[int, int] | list[int] | None = None
    num_bjet_range: tuple[int, int] | list[int] | None = None
    max_num_events: int | None = None
    random_seed: int | None = 31415
    group_indices: list[list[int]] = field(
        default_factory=lambda: [[0, 1, 2], [3, 4, 5]]
    )

    def __post_init__(self):
        if isinstance(self.matchability_criteria, int):
            self.matchability_criteria = [
                self.matchability_criteria,
            ]
        assert isinstance(self.matchability_criteria, list), (
            "matchability_criteria must be an int or a list of ints"
        )


@dataclass
class HyperH5Structure:
    """Structure of H5 file for HyPER datasets."""

    feature_key: str = "jets"
    matchability_key: str = "matchability"
    njet_count_key: str = "njet"
    nbjet_count_key: str = "nbjet"
    truthmatch_key: str = "jet_match"
    lengths_key: str = field(init=False)

    def __post_init__(self):
        self.lengths_key = self.njet_count_key


@dataclass
class HyperDatasetConfig:
    """Configuration for HyPER datasets.

    Attributes:
        features:
            Sequence of transformations to apply to the dataset features.
        structure:
            Structure of the HDF5 file, initialized to None by default.
        matchability_criteria:
            This is notably different from Topograph. The equation for matchability in
            HyPER dataset is: sum(2**(jet_truthmatch - 1)) for all jets in event.
            Given that the ordering of jet_truthmatch is "inverted" compared to
            Topograph. I.e. 0bxyzuvw in HyPER corresponds to 0bwvuxyz in Topograph.
            See https://zenodo.org/records/10653837 for more details.
        num_jet_range:
            Tuple specifying the (min, max) number of required jets. Both ends are
            inclusive.
        num_bjet_range:
            Tuple specifying the (min, max) number of required b-jets. Both ends are
            inclusive.
        chunk_size:
            Data will be read in chunks, when processing the dataset. This parameter
            doesn't affect the final dataset.
        pad_length:
            Length to pad the sequences to.
        max_mask_level:
            Maximum level of masking to apply to the adjacency matrix. The value is
            included. Mask level 0: Full masking, level 1: Single visible W, level 2:
            Both W's visible, level 3: One top fully visible and visible W.
        max_num_events:
            Maximum number of events to load from the dataset, can be set to None to
            load all events.

        use_masking: bool
            If False, no masking will be applied to the adjacency matrix. This is
            required in the case of loading partial events for inference. But for
            training diffusion models, masking is necessary.

    """

    features: Sequence[BaseTransform]
    structure: HyperH5Structure = field(default_factory=HyperH5Structure)
    num_jet_range: tuple[int, int] | list[int] | None = None
    num_bjet_range: tuple[int, int] | list[int] | None = None
    matchability_criteria: int | list[int] = 0b111111
    chunk_size: int = 100_000
    pad_length: int = 20
    max_mask_level: int = 3
    max_num_events: int | None = None
    use_masking: bool = True

    min_allowed_njets: int | None = field(init=False)
    max_allowed_njets: int | None = field(init=False)
    min_allowed_nbjets: int | None = field(init=False)
    max_allowed_nbjets: int | None = field(init=False)

    def __post_init__(self):
        if exists(self.num_jet_range):
            self.min_allowed_njets, self.max_allowed_njets = self.num_jet_range
        else:
            self.min_allowed_njets = None
            self.max_allowed_njets = None
        if exists(self.num_bjet_range):
            self.min_allowed_nbjets, self.max_allowed_nbjets = self.num_bjet_range
        else:
            self.min_allowed_nbjets = None
            self.max_allowed_nbjets = None

        if isinstance(self.matchability_criteria, int):
            self.matchability_criteria = [
                self.matchability_criteria,
            ]
        assert isinstance(self.matchability_criteria, list), (
            "matchability_criteria must be an int or a list of ints"
        )


@dataclass
class DataPaths:
    """Paths to the dataset files.

    Attributes:
        train_path:
            Path to the training dataset file.
        test_path:
            Path to the test dataset file.
        predict_path:
            Optional path for prediction dataset file, defaults to test_path if not
                provided.
    """

    train_path: Path
    val: float | Path
    test_path: Path
    predict_path: Path | None = None

    def __post_init__(self):
        self.train_path = Path(self.train_path)
        assert isinstance(self.train_path, PathLike)
        self.test_path = Path(self.test_path)
        assert isinstance(self.test_path, PathLike)
        if self.predict_path is None:
            self.predict_path = Path(self.test_path)
        assert isinstance(self.predict_path, PathLike)

    @property
    def val_frac(self) -> float:
        """The fraction of the dataset to use for validation."""
        return self.val if isinstance(self.val, float) else 1.0

    @property
    def train_frac(self) -> float:
        """The fraction of the dataset to use for training."""
        return 1.0 - self.val if isinstance(self.val, float) else 1.0

    @property
    def val_path(self) -> PathLike:
        """The path to the validation dataset."""
        return self.val if isinstance(self.val, PathLike) else self.train_path


@dataclass
class DataLoaderConfig:
    """Configuration for data loaders.

    Attributes:
        batch_size:
            Number of samples per batch.
        num_workers:
            Number of subprocesses to use for data loading.
        shuffle:
            Whether to shuffle the dataset at every epoch.
        drop_last:
            Whether to drop the last incomplete batch if the dataset size is not
                divisible by the batch size.
        pin_memory:
            Whether to copy tensors into CUDA pinned memory for faster transfer to GPU.
        prefetch_factor:
            Number of samples loaded in advance by each worker.
        length_splits:
            Optional list of lengths to split the dataset into buckets for efficient
            batching. If provided, batches will be formed from samples of similar
            lengths to minimize padding.
    """

    batch_size: int
    num_workers: int
    shuffle: bool = False
    drop_last: bool = False
    pin_memory: bool = True
    prefetch_factor: int = 2
    length_splits: list[int] | None = None


def collate_batch(batches: list[SequenceBatch]) -> SequenceBatches:
    max_size = max(batch["features"].shape[0] for batch in batches)
    feature_names = batches[0]["feature_names"]
    out_batch = default_collate(
        [
            {
                "features": th.nn.functional.pad(
                    batch["features"], (0, 0, 0, max_size - batch["features"].shape[0])
                ),
                "mask": th.nn.functional.pad(
                    batch["mask"], (0, max_size - batch["mask"].shape[0])
                ),
                "groups": batch["groups"],
                "adjacency": th.nn.functional.pad(
                    batch["adjacency"],
                    (
                        0,
                        left_over_size := max_size - batch["adjacency"].shape[0],
                        0,
                        left_over_size,
                    ),
                ),
                "w_adjacency": th.nn.functional.pad(
                    batch["w_adjacency"],
                    (
                        0,
                        left_over_size := max_size - batch["w_adjacency"].shape[0],
                        0,
                        left_over_size,
                    ),
                ),
                "w_adjacency1": th.nn.functional.pad(
                    batch["w_adjacency1"],
                    (
                        0,
                        left_over_size := max_size - batch["w_adjacency1"].shape[0],
                        0,
                        left_over_size,
                    ),
                ),
                "w_adjacency2": th.nn.functional.pad(
                    batch["w_adjacency2"],
                    (
                        0,
                        left_over_size := max_size - batch["w_adjacency2"].shape[0],
                        0,
                        left_over_size,
                    ),
                ),
                "top_adjacency1": th.nn.functional.pad(
                    batch["top_adjacency1"],
                    (
                        0,
                        left_over_size := max_size - batch["top_adjacency1"].shape[0],
                        0,
                        left_over_size,
                    ),
                ),
                "top_adjacency2": th.nn.functional.pad(
                    batch["top_adjacency2"],
                    (
                        0,
                        left_over_size := max_size - batch["top_adjacency2"].shape[0],
                        0,
                        left_over_size,
                    ),
                ),
                "mask_adjacency": th.nn.functional.pad(
                    batch["mask_adjacency"],
                    (
                        0,
                        0,
                        0,
                        left_over_size := max_size - batch["mask_adjacency"].shape[0],
                        0,
                        left_over_size,
                    ),
                    value=True,
                ),
                "mask_level": batch["mask_level"],
            }
            for batch in batches
        ]
    )
    out_batch["feature_names"] = feature_names
    return cast("SequenceBatches", out_batch)
