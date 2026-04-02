import torch as th

from ml_utils.components.transformer import (
    BiasTransformerEncoder,
    BiasTransformerEncoderConfig,
)
from ml_utils.utils import default

from .base import SequenceWrapperModel


class BiasTransformerWrapper(SequenceWrapperModel):
    def __init__(
        self,
        dim_single: int,
        dim_pair: int,
        config: BiasTransformerEncoderConfig | None = None,
    ):
        super().__init__()
        config = default(config, BiasTransformerEncoderConfig())
        self._dim_single = dim_single
        self._dim_pair = dim_pair
        self._model = BiasTransformerEncoder(
            in_features=dim_single,
            bias_features=dim_pair,
            config=config,
        )

    @property
    def dim_single(self) -> int:
        return self._dim_single

    @property
    def dim_pair(self) -> int:
        return self._dim_pair

    def forward(
        self, single_features: th.Tensor, pair_features: th.Tensor, mask: th.Tensor
    ) -> tuple[
        th.Tensor,
        th.Tensor,
        th.Tensor,
    ]:
        out_single = self._model(
            single_features,
            bias=pair_features,
            mask=mask,
        )
        # Pair features are static.
        return out_single, pair_features, mask
