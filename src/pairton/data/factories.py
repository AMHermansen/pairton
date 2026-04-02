from abc import ABC, abstractmethod
from pathlib import Path
from typing import override

from torch.utils.data import Dataset

from pairton.data.hyper import HyperDataset
from pairton.data.types import SequenceBatch
from pairton.data.utils import (
    HyperDatasetConfig,
)


class DatasetFactory(ABC):
    @abstractmethod
    def create_sequence_dataset(self, data_path: Path) -> Dataset[SequenceBatch]:
        """Method to create a sequence dataset.

        Args:
            data_path: Path to the dataset.

        Returns:
            A Dataset object containing sequence batches.
        """

    @property
    @abstractmethod
    def num_features(self) -> int:
        pass


class HyperDatasetFactory(DatasetFactory):
    def __init__(
        self,
        config: HyperDatasetConfig,
    ):
        """Construct a Dataset Factory using the Hyper backend.

        Args:
            config: Configuration for the Hyper dataset.
        """
        self._config = config

    @override
    def create_sequence_dataset(self, data_path: Path) -> Dataset[SequenceBatch]:
        return HyperDataset(data_path=data_path, config=self._config)

    @property
    @override
    def num_features(self) -> int:
        return len(self._config.features)
