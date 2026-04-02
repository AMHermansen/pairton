from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import replace
from os import PathLike
from pathlib import Path
from typing import Any, Literal, cast, override

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from pairton.data.types import SequenceBatch
from pairton.data.factories import DatasetFactory
from pairton.data.utils import (
    DataLoaderConfig,
    DataPaths,
    collate_batch,
)
from ml_utils.data_utils import SequenceBucketingSampler
from ml_utils.utils import exists


class BaseDataModule[T](LightningDataModule, ABC):
    def __init__(
        self,
        *,
        data_paths: DataPaths,
        data_loader_config: DataLoaderConfig,
        validation_override: dict[str, Any],
        dataset_factory: DatasetFactory,
    ):
        """Constructs a data module for the ttbar dataset.

        Args:
            data_paths: Paths to the dataset files for different modes.
            data_loader_config: Configuration for the DataLoader.
            validation_override: Overrides for the DataLoader configuration during
                validation.
            dataset_factory: Factory to create dataset instances.
        """
        super().__init__()
        # Initialize datasets to None
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self._predict_dataset = None

        self.data_loader_config = data_loader_config
        self.data_paths = data_paths
        self._validation_override = validation_override

        self._train_frac = 1 - self.data_paths.val_frac
        self._val_frac = self.data_paths.val_frac

        self._dataset_factory = dataset_factory

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        self._dataset = self._make_dataset(stage)
        if stage == "fit":
            if isinstance(self.data_paths.val, PathLike):
                self._train_dataset = self._make_dataset(stage)
                self._val_dataset = self._make_dataset("validate")
            else:
                self._train_dataset, self._val_dataset = random_split(
                    self._dataset,
                    [
                        int(len(self._dataset) * self._train_frac),
                        int(len(self._dataset) * self._val_frac),
                    ],
                )
        elif stage == "validate":
            self._val_dataset = self._dataset
        elif stage == "test":
            self._test_dataset = self._dataset
        elif stage == "predict":
            self._predict_dataset = self._dataset

    @abstractmethod
    def _make_dataset(
        self, stage: Literal["fit", "validate", "test", "predict"]
    ) -> Dataset[T]:
        """Creates a dataset for the given mode.

        Args:
            stage: The mode for which to create the dataset

        Returns:
            A Dataset instance for the specified mode.
        """

    @property
    @abstractmethod
    def collate_fn(self) -> Callable[[Any], Any]:
        """Returns the collate function for the DataLoader."""

    def _create_sampler(
        self,
        dataset: Dataset[T],
        config: DataLoaderConfig,
    ) -> SequenceBucketingSampler | None:
        """Creates a sampler for the given dataset and DataLoader configuration.

        Args:
            dataset: The dataset for which to create the sampler.
            config: The DataLoader configuration.

        Returns:
            A SequenceBucketingSampler instance for the specified dataset and configuration.
        """
        try:
            lengths = dataset.lengths  # type: ignore[attr-defined]
        except AttributeError:
            return None
        if not exists(config.length_splits):
            return None
        return SequenceBucketingSampler(
            lengths=lengths,
            batch_size=config.batch_size,
            drop_exceeding=config.drop_last,
            shuffle=config.shuffle,
            length_splits=config.length_splits,
        )

    def _make_dataloader(self, dataset: Dataset[T], mode: str) -> DataLoader[T]:
        """Creates a DataLoader for the given dataset and mode.

        Args:
            dataset: The dataset to create a DataLoader for.
            mode: The mode for which to create the DataLoader.

        Returns:
            A DataLoader instance for the specified dataset and mode.
        """
        if mode == "train":
            config = self.data_loader_config
        else:
            config = replace(self.data_loader_config, **self._validation_override)
        sampler = self._create_sampler(dataset, config)
        if sampler is not None:
            config = replace(
                config,
                shuffle=False,
                drop_last=False,
            )
            return DataLoader(
                dataset=dataset,
                collate_fn=self.collate_fn,
                batch_sampler=sampler,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                prefetch_factor=config.prefetch_factor,
            )
        return DataLoader(
            dataset=dataset,
            collate_fn=self.collate_fn,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            drop_last=config.drop_last,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
        )

    def train_dataloader(self) -> DataLoader[T]:
        assert self._train_dataset is not None, "Train dataset is not set up."
        return self._make_dataloader(self._train_dataset, "train")

    def val_dataloader(self) -> DataLoader[T]:
        assert self._val_dataset is not None, "Validation dataset is not set up."
        return self._make_dataloader(self._val_dataset, "validate")

    def test_dataloader(self) -> DataLoader[T]:
        assert self._test_dataset is not None, "Test dataset is not set up."
        return self._make_dataloader(self._test_dataset, "test")

    def predict_dataloader(self) -> DataLoader[T]:
        assert self._predict_dataset is not None, "Predict dataset is not set up."
        return self._make_dataloader(self._predict_dataset, "predict")

    def _get_datapath(
        self, stage: Literal["fit", "validate", "test", "predict"]
    ) -> Path:
        if stage == "fit":
            return Path(self.data_paths.train_path)
        if stage == "validate":
            if isinstance(self.data_paths.val, PathLike):
                return Path(self.data_paths.val)
            return Path(self.data_paths.train_path)
        if stage == "test":
            return Path(self.data_paths.test_path)
        if stage == "predict":
            return Path(self.data_paths.predict_path)
        raise ValueError(f"Unknown mode: {stage}")

    def _get_path(self, mode: Literal["train", "validate", "test", "predict"]):
        if mode == "train":
            path = self.data_paths.train_path
        elif mode == "validate":
            path = self.data_paths.val_path
        elif mode == "test":
            path = self.data_paths.test_path
        elif mode == "predict":
            path = self.data_paths.predict_path
        else:
            raise ValueError(f"Unknown mode: {mode}")
        assert isinstance(path, PathLike), (
            "Predict path should've been PathLike by now."
        )
        return path


class SequenceDataModule(BaseDataModule):
    @override
    def _make_dataset(
        self, mode: Literal["fit", "validate", "test", "predict"]
    ) -> Dataset[SequenceBatch]:
        """Creates a dataset for the given mode.

        Args:
            mode: The mode for which to create the dataset

        Returns:
            A TTBarDataset instance for the specified mode.
        """
        stage = "train" if mode == "fit" else mode
        stage = cast("Literal['train', 'validate', 'test', 'predict']", stage)
        path = self._get_path(stage)
        return self._dataset_factory.create_sequence_dataset(
            data_path=path,
        )

    @property
    @override
    def collate_fn(self) -> Callable[[Any], Any]:
        """Returns the collate function for the DataLoader."""
        return collate_batch
