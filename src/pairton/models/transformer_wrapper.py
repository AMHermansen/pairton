import torch as th

from ml_utils.components import TransformerEncoder, TransformerEncoderConfig
from ml_utils.torch_utils import pack_tensor, unpack_tensor
from ml_utils.utils import default

from .base import SequenceWrapperModel


class TransformerWrapper(SequenceWrapperModel):
    def __init__(
        self,
        dim_single: int,
        config: TransformerEncoderConfig | None = None,
    ):
        super().__init__()
        self._config = default(config, TransformerEncoderConfig())
        self._dim_single = dim_single

        self._model = TransformerEncoder(in_features=dim_single, config=self._config)

    @property
    def dim_single(self) -> int:
        return self._model.in_features

    @property
    def dim_pair(self) -> None:
        return None

    def forward(
        self, single_features: th.Tensor, pair_features: th.Tensor, mask: th.Tensor,
    ) -> tuple[th.Tensor, None, th.Tensor]:
        cu_seqlens, packed_single_features = pack_tensor(mask, single_features)
        packed_output, updated_cu_seqlens, _ = self._model(packed_single_features, cu_seqlens)
        mask, output = unpack_tensor(updated_cu_seqlens, packed_output)
        return output, None, mask
