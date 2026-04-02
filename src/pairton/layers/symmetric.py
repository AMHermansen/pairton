import einx
import jaxtyping as jx
import torch as th

from torch import nn


class MultiDimensionalLinearNoBiasThenOuterSum(nn.Module):
    def __init__(
        self,
        dim: int,
        repeats: list[int],
        dim_out: int | None = None,
    ):
        """Applies linear no bias followed by outer sum over multiple dimensions.

        Args:
            dim: Input dimension.
            repeats: List of integers specifying the number of symmetric indices.
            dim_out: Output dimension. If None, defaults to input dimension.
        """
        super().__init__()
        self.dim = dim
        self.repeats = repeats
        self.dim_out = dim_out if dim_out is not None else dim
        self._linears = nn.ModuleList(
            [nn.Linear(dim, self.dim_out, bias=False) for _ in repeats]
        )
        self._einx_format = self._get_einx_format(repeats)

    @staticmethod
    def _get_einx_format(repeats) -> str:
        initial_parts = []
        for idx in range(sum(repeats)):
            initial_parts.append(f"b i{idx} d")
        einsum_input = ", ".join(initial_parts)
        final_part = "b " + " ".join([f"i{idx}" for idx in range(sum(repeats))]) + " d"
        return f"{einsum_input} -> {final_part}"

    def forward(
        self,
        t: jx.Float[th.Tensor, "batch seq_len dim"],
    ) -> th.Tensor:
        """Forward pass.

        Args:
            t: Input tensor of shape (batch, seq_len, dim).

        Returns:
            Output tensor after applying linear no bias and outer sum.
            Shape will be (batch, seq_len, seq_len, ..., dim_out) with the number of
            seq_len dimensions equal to sum(repeats).
        """
        embeddings = [
            linear(t)
            for linear, count in zip(self._linears, self.repeats)
            for _ in range(count)
        ]
        return einx.add(self._einx_format, *embeddings)
