from pathlib import Path

import h5py
import numpy as np
import torch as th
from torch.utils.data import Dataset

from pairton.data.types import SequenceBatch
from pairton.data.utils import HyperDatasetConfig
from ml_utils.np_utils.packing import unpack_array
from ml_utils.utils import exists

from .utils import construct_adjacency, mask_adjacency


class HyperDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        config: HyperDatasetConfig,
    ):
        """Construct dataset from HyPER structure h5 file.

        Args:
            data_path: Path to the HyPER h5 file.
            config: HyperDatasetConfig object containing dataset configuration.
        """
        self._config = config
        self._data_path = data_path
        self._setup()

    def __getitem__(self, idx: int) -> SequenceBatch:
        if idx >= len(self):
            raise IndexError("Index out of bounds")

        start_pos = self._cu_seqlens[idx]
        end_pos = self._cu_seqlens[idx + 1]
        features = self._features[start_pos:end_pos]
        mask = th.zeros(self._config.pad_length, dtype=th.bool)

        mask[: len(features)] = True
        padded_features = np.zeros(
            (self._config.pad_length, features.shape[1]), dtype=features.dtype
        )
        padded_features[: len(features), :] = features
        matched_indices = self._truth_match[start_pos:end_pos]
        truth_groups = self._get_truth_groups(matched_indices)
        groups = [th.Tensor(truth_group).to(th.int32) for truth_group in truth_groups]

        mask_level = th.randint(0, self._config.max_mask_level + 1, size=(1,)).item()
        return SequenceBatch(
            features=th.Tensor(features),
            mask=mask,
            groups=groups,
            adjacency=construct_adjacency(
                th.Tensor(features),
                [th.Tensor(group).to(th.int32) for group in truth_groups],
            ),
            w_adjacency=construct_adjacency(
                th.Tensor(features),
                [th.Tensor(group).to(th.int32)[1:] for group in truth_groups],
            ),
            w_adjacency1=construct_adjacency(
                th.Tensor(features), [th.Tensor(truth_groups[0]).to(th.int32)[1:]]
            ),
            w_adjacency2=construct_adjacency(
                th.Tensor(features), [th.Tensor(truth_groups[1]).to(th.int32)[1:]]
            ),
            top_adjacency1=construct_adjacency(
                th.Tensor(features), [th.Tensor(truth_groups[0]).to(th.int32)]
            ),
            top_adjacency2=construct_adjacency(
                th.Tensor(features), [th.Tensor(truth_groups[1]).to(th.int32)]
            ),
            mask_adjacency=mask_adjacency(
                num_jets=th.sum(mask).item(),
                mask_level=mask_level,
                groups=groups,
            )
            if self._config.use_masking
            else th.zeros(len(features), len(features), 2, dtype=th.bool),
            feature_names=[
                transform.out_feature_name for transform in self._config.features
            ],
            mask_level=mask_level,
        )

    def __len__(self) -> int:
        return len(self._cu_seqlens) - 1

    @property
    def lengths(self) -> np.ndarray:
        return self._lengths

    def _setup(self):
        with h5py.File(self._data_path, "r") as hf:
            self._cu_seqlens = self._load_cu_seqlens(
                hf,
                self._config.structure.lengths_key,
            )
            matchability_dataset = hf[self._config.structure.matchability_key]
            assert isinstance(matchability_dataset, h5py.Dataset)
            matchability_mask = self._get_matchability_mask(
                matchability_dataset,
                self._config.matchability_criteria,
            )
            num_jet_dataset = hf[self._config.structure.njet_count_key]
            assert isinstance(num_jet_dataset, h5py.Dataset)
            njet_mask = self._get_mask_from_counts(
                num_jet_dataset,
                self._config.min_allowed_njets,
                self._config.max_allowed_njets,
            ).squeeze()
            num_bjet_dataset = hf[self._config.structure.nbjet_count_key]
            assert isinstance(num_bjet_dataset, h5py.Dataset)
            nbjet_mask = self._get_mask_from_counts(
                num_bjet_dataset,
                self._config.min_allowed_nbjets,
                self._config.max_allowed_nbjets,
            ).squeeze()
            combined_mask = np.logical_and.reduce([
                matchability_mask,
                njet_mask,
                nbjet_mask,
            ])
            combined_mask = self._maybe_restrict_max_events(
                combined_mask,
                self._config.max_num_events,
            )

            features = np.concatenate([
                self._mask_packed_data(
                    self._load_chunk_data(
                        hf[self._config.structure.feature_key],
                        self._cu_seqlens,
                        i,
                        i + self._config.chunk_size,
                    ),
                    self._cu_seqlens,
                    combined_mask,
                    i,
                    i + self._config.chunk_size,
                )
                for i in range(0, len(self._cu_seqlens) - 1, self._config.chunk_size)
            ])
            feature_names = [
                x.decode()
                for x in hf[f"{self._config.structure.feature_key}_variables"]
            ]
            self._features = self._process_features(
                features,
                feature_names,
            )
            self._truth_match = self._load_jet_match(
                hf[self._config.structure.truthmatch_key],
                self._cu_seqlens,
                combined_mask,
            )
            lengths = np.diff(self._cu_seqlens, axis=0)
            self._cu_seqlens = np.concatenate([
                np.array([0]),
                lengths[combined_mask].cumsum(),
            ])
            self._lengths = np.diff(self._cu_seqlens, axis=0)

    def _load_jet_match(
        self,
        h5_dataset: h5py.Dataset,
        cu_seqlens: np.ndarray,
        combined_mask: np.ndarray,
    ):
        truth_match_data = h5_dataset[:]
        return self._mask_packed_data(
            truth_match_data,
            cu_seqlens,
            combined_mask,
            0,
            len(cu_seqlens) - 1,
        ).squeeze()

    @staticmethod
    def _mask_packed_data(
        chunk_data: np.ndarray,
        cu_seqlens: np.ndarray,
        combined_mask: np.ndarray,
        start_idx: int,
        end_idx: int,
    ) -> np.ndarray:
        """Filter chunk data based on provided masks.

        Args:
            chunk_data: Data for the specific chunk.
            cu_seqlens: Cumulative sequence lengths array.
            combined_mask: Combined boolean mask for filtering.
            start_idx: Start index of the chunk.
            end_idx: End index of the chunk.
        """
        reduced_combined_mask = combined_mask[start_idx:end_idx]
        reduced_cu_seqlens = cu_seqlens[start_idx : end_idx + 1]
        chunk_mask, _ = unpack_array(reduced_cu_seqlens, chunk_data)
        filtered_chunk_mask = np.zeros_like(chunk_mask, dtype=bool)
        filtered_chunk_mask[reduced_combined_mask, :] = True
        filtered_chunk_mask = filtered_chunk_mask[chunk_mask]
        return chunk_data[filtered_chunk_mask]

    @staticmethod
    def _load_chunk_data(
        h5_dataset: h5py.Dataset,
        cu_seqlens: np.ndarray,
        start_idx: int,
        end_idx: int,
    ) -> np.ndarray:
        """Load data for a specific chunk index.

        Args:
            cu_seqlens: Cumulative sequence lengths array.
            start_idx: Start index of the chunk.
            end_idx: End index of the chunk.
        """
        start_pos = cu_seqlens[start_idx]
        end_pos = cu_seqlens[min(end_idx, len(cu_seqlens) - 1)]
        return h5_dataset[start_pos:end_pos]

    @staticmethod
    def _load_cu_seqlens(
        h5_file: h5py.File,
        lengths_key: str,
    ):
        lengths_dataset = h5_file[lengths_key]
        assert isinstance(lengths_dataset, h5py.Dataset)
        lengths = lengths_dataset[:]
        cu_seqlens = np.zeros(len(lengths) + 1, dtype=lengths.dtype)
        np.cumsum(lengths, out=cu_seqlens[1:])
        return cu_seqlens

    @staticmethod
    def _get_matchability_mask(
        matchability_data: h5py.Dataset,
        allowed_matchability_values: list[int],
    ):
        matchability_values = matchability_data[:]

        # Mask corresponding to OR_i[(X_i && v) == v].
        # Effectively checks if v-bitmask is contained in X.
        return np.logical_or.reduce([
            np.equal(
                np.bitwise_and(
                    matchability_values,
                    matchability_value,
                ),
                matchability_value,
            )
            for matchability_value in allowed_matchability_values
        ])

    @staticmethod
    def _get_mask_from_counts(
        count_dataset: h5py.Dataset,
        min_allowed_jets: int | None = None,
        max_allowed_jets: int | None = None,
    ):
        num_jets_values = count_dataset[:]
        mask = np.ones_like(num_jets_values, dtype=bool)

        if exists(min_allowed_jets):
            mask &= num_jets_values >= min_allowed_jets
        if exists(max_allowed_jets):
            mask &= num_jets_values <= max_allowed_jets
        return mask

    @staticmethod
    def _get_truth_groups(matched_indices: np.ndarray) -> list[np.ndarray]:
        matched_idx_sorted = np.argsort(-matched_indices)
        idx = -np.ones(6, dtype=np.int32)
        for pos in matched_idx_sorted[:6]:
            if matched_indices[pos] == 0:
                break
            idx[matched_indices[pos] - 1] = pos
        return [idx[:3], idx[3:]]

    @staticmethod
    def _maybe_restrict_max_events(
        combined_mask: np.ndarray,
        max_num_events: int | None,
    ) -> np.ndarray:
        if exists(max_num_events):
            true_indices = np.where(combined_mask)[0]
            # Randomly select indices to keep
            if len(true_indices) > max_num_events:
                selected_indices = np.random.choice(
                    true_indices,
                    size=max_num_events,
                    replace=False,
                )
                new_mask = np.zeros_like(combined_mask, dtype=bool)
                new_mask[selected_indices] = True
                combined_mask[:] = new_mask
        return combined_mask

    def _process_features(
        self,
        features: np.ndarray,
        feature_names: list[str],
    ) -> np.ndarray:
        return np.stack(
            [
                transform(features[:, feature_names.index(transform.in_feature_name)])
                for transform in self._config.features
            ],
            axis=-1,
        )
