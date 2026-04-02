from abc import ABC, abstractmethod

import torch as th
from torch import nn


class SequenceWrapperModel(nn.Module, ABC):
    @property
    @abstractmethod
    def dim_single(self) -> int:
        """Dimension of a single element in the sequence."""

    @property
    @abstractmethod
    def dim_pair(self) -> int | None:
        """Dimension of a pair of elements in the sequence."""

    @abstractmethod
    def forward(
        self,
        single_features: th.Tensor,
        pair_features: th.Tensor,
        mask: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor | None, th.Tensor,]:
        """Forward pass of the model.

        Args:
            single_features (th.Tensor): Tensor of shape (N, L, D_single) representing single element features.
            pair_features (th.Tensor): Tensor of shape (B, L, L, D_pair) representing pairwise element features.
            mask (th.Tensor): Tensor of shape (N, L) representing the valid elements in the sequence.

        Returns:
            tuple of two tensors:
                - th.Tensor: Processed single features of shape (N, L1, D_single).
                - th.Tensor or None: Processed pair features of shape (B, L1, L1, D_pair) if dim_pair is not None, else None.
                - th.Tensor: Corresponding mask of shape (N, L1).
        """
