# Unit test for hyperedge softmax operation defined in HyPER and re-implemntation

# ==================== HyPER License ===================================
# HyPER (https://github.com/tzuhanchang/HyPER)
# MIT License
#
# Copyright (c) 2024 Tzu-Han Chang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch as th
from torch import Tensor
from torch_geometric.utils import scatter
import einx

# reimplemented version
from pairton.layers.hyper import hyper_softmax as hyper_softmax_re


# Implementation from HyPER (https://github.com/tzuhanchang/HyPER)
def hyper_softmax(
    src: Tensor,
    index: Tensor,
    dim_size: int,
    dim: int = 0,
) -> Tensor:
    r"""This function is a modified version of `torch_geometric.utils.softmax`,
    which solves value unconstrain error raised by `torch.onnx.dynamo_export`.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the
            softmax.
        dim_size (int): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned.
        dim (int, optional): The dimension in which to normalize.
            (default: :obj:`0`)

    :rtype: :class:`Tensor`
    """
    src_max = scatter(src.detach(), index, dim, dim_size=dim_size, reduce="max")
    out = src - src_max.index_select(dim, index)
    out = out.exp()
    out_sum = scatter(out, index, dim, dim_size=dim_size, reduce="sum") + 1e-16
    out_sum = out_sum.index_select(dim, index)

    return out / out_sum


def test_hyper_softmax():
    mask = th.tensor(
        [
            [True, True, True, True, True, True, False, False],
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, False],
        ]
    )
    n_nodes = mask.sum(-1)
    n_hyper_edges = n_nodes * (n_nodes - 1) * (n_nodes - 2) / 6
    hyper_edge_idx = th.repeat_interleave(
        th.arange(mask.shape[0], dtype=th.int64), n_hyper_edges.long()
    )

    features = th.randn(mask.shape[0], mask.shape[1], 16)
    full_hyper_edge_features = th.einsum(
        "bic,bjc,bkc->bijkc", features, features, features
    )
    hyper_edge_mask = einx.logical_and("b i, b j, b k -> b i j k", mask, mask, mask)
    max_seq_len = mask.shape[1]
    node_idx = th.arange(max_seq_len, device=mask.device)
    i = node_idx.view(1, -1, 1, 1)
    j = node_idx.view(1, 1, -1, 1)
    k = node_idx.view(1, 1, 1, -1)
    ijk_valid = (i < j) & (j < k)
    hyper_edge_mask = hyper_edge_mask & ijk_valid

    ref_out = hyper_softmax(
        full_hyper_edge_features[hyper_edge_mask], hyper_edge_idx, None, dim=0
    )

    custom_out = hyper_softmax_re(mask, full_hyper_edge_features)
    th.testing.assert_close(custom_out[hyper_edge_mask.flatten(1)], ref_out)


if __name__ == "__main__":
    test_hyper_softmax()
