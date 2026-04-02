from typing import override

import torch as th

from ml_utils.components.pairformer import PairFormer, PairFormerConfig
from ml_utils.utils import default

from .base import SequenceWrapperModel


class PairFormerWrapper(SequenceWrapperModel):
    def __init__(
        self,
        dim_single: int,
        dim_pair: int,
        config: PairFormerConfig | None = None,
    ):
        """Initialize PairFormerWrapper.

        Args:
            dim_single: single feature dimension
            dim_pair: pair feature dimension
            config: PairFormer configuration
        """
        super().__init__()
        self._dim_single = dim_single
        self._dim_pair = dim_pair
        self._config = default(config, PairFormerConfig())
        self._model = PairFormer(
            single_features=dim_single,
            pair_features=dim_pair,
            config=config,
        )

    @property
    @override
    def dim_single(self) -> int:
        return self._dim_single

    @property
    @override
    def dim_pair(self) -> int:
        return self._dim_pair

    @override
    def forward(
        self,
        single_features: th.Tensor,
        pair_features: th.Tensor,
        mask: th.Tensor,
    ) -> tuple[
        th.Tensor,
        th.Tensor,
        th.Tensor,
    ]:
        # Minimal wrapping around PairFormer to adapt to SequenceWrapperModel interface
        seq_lens = mask.sum(dim=-1)

        embedded_single_features, embedded_pair_features = self._model(
            single_features=single_features,
            pair_features=pair_features,
            seq_lens=seq_lens,
            mask=mask,
        )
        return embedded_single_features, embedded_pair_features, mask
