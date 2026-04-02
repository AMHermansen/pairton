import einx
import torch as th
from torch import nn
from ml_utils.components import MLP, MLPConfig
from ml_utils.utils import default


def hyper_softmax(mask, x):
    hyper_edge_mask = einx.logical_and("b i, b j, b k -> b i j k", mask, mask, mask)
    max_seq_len = mask.shape[1]
    node_idx = th.arange(max_seq_len, device=mask.device)
    i = node_idx.view(1, -1, 1, 1)
    j = node_idx.view(1, 1, -1, 1)
    k = node_idx.view(1, 1, 1, -1)
    ijk_valid = (i < j) & (j < k)
    hyper_edge_mask = hyper_edge_mask & ijk_valid
    a_flat = th.softmax(
        x.masked_fill(~hyper_edge_mask.unsqueeze(-1), -th.inf).flatten(1, 3),
        dim=1,
    )
    return a_flat


class HyperEdgeLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 32,
        config_inner: MLPConfig | None = None,
        config_outer: MLPConfig | None = None,
    ):
        super().__init__()
        config_inner = default(config_inner, MLPConfig())
        config_outer = default(config_outer, MLPConfig())
        self._mlp_outer = MLP(2 * hidden_features, 1, config=config_outer)  # logits out
        self._inner_embed = nn.Linear(in_features, in_features, bias=False)
        self._mlp_inner = MLP(in_features, hidden_features, config=config_inner)

    def forward(self, x: th.Tensor, mask: th.Tensor):
        """Args:
            x: Shape (batch_size, num_nodes, num_nodes, num_nodes, in_features)
            mask: Shape (batch_size, num_nodes) Cumulative sequence lengths.

        Returns:
            out: Shape (batch_size, num_nodes, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _, _, in_features = x.shape
        x_embed = self._mlp_inner(x)
        a_flat = hyper_softmax(mask, x_embed)
        x_flat_inner = th.relu(self._inner_embed(x_embed.flatten(1, 3)))
        return self._mlp(
            th.cat(
                [
                    a_flat * x_flat_inner,
                    x_embed.flatten(1, 3),
                ],
                dim=-1,
            )
        ).reshape(batch_size, num_nodes, num_nodes, num_nodes)
