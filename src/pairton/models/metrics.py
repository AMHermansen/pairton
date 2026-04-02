from typing import Any, override

import jaxtyping as jx
import torch as th
from torchmetrics import Metric

from ml_utils.utils import default, exists


def compute_is_valid(
    current_valid: jx.Bool[th.Tensor, "batch"],
    mask: jx.Bool[th.Tensor, "batch seq_len"],
    njet_range: tuple[int, int] | None,
):
    if exists(njet_range):
        njets = th.sum(mask, dim=1)
        is_in_njet_range = th.logical_and(
            njets >= njet_range[0],
            njets <= njet_range[1],
        )
        return th.logical_and(current_valid, is_in_njet_range)
    return current_valid


class DiffusionFullTPR(Metric):
    def __init__(
        self,
        njet_range: tuple[int, int] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._njet_range = njet_range
        self.add_state("correct", default=th.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=th.tensor(0), dist_reduce_fx="sum")

    @override
    def update(
        self,
        t_adjacency_guess: jx.Int[th.Tensor, "batch seq_len seq_len"],
        w_adjacency_guess: jx.Int[th.Tensor, "batch seq_len seq_len"],
        t_adjacency_target: jx.Int[th.Tensor, "batch seq_len seq_len"],
        w_adjacency_target: jx.Int[th.Tensor, "batch seq_len seq_len"],
        mask: jx.Bool[th.Tensor, "batch seq_len"],
        groups: list[jx.Int[th.Tensor, "n_groups"]],
    ):
        is_full_group = th.logical_and(
            th.all(groups[0] >= 0, dim=-1),
            th.all(groups[1] >= 0, dim=-1),
        )
        is_valid = compute_is_valid(is_full_group, mask, self._njet_range)
        batch_size, max_seq_len, _ = t_adjacency_guess.shape
        seq_lens = th.sum(mask, dim=1)
        count_row = (
            th.arange(max_seq_len, device=mask.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, max_seq_len, 1)
        )
        count_col = (
            th.arange(max_seq_len, device=mask.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(batch_size, 1, max_seq_len)
        )
        out_of_bounds_mask = (
            (count_row >= seq_lens.unsqueeze(-1).unsqueeze(-1))
            | (count_col >= seq_lens.unsqueeze(-1).unsqueeze(-1))
        ).to(th.bool)
        masked_t_adjacency_guess = t_adjacency_guess.masked_fill(out_of_bounds_mask, 0)
        masked_w_adjacency_guess = w_adjacency_guess.masked_fill(out_of_bounds_mask, 0)
        masked_t_adjacency_target = t_adjacency_target.masked_fill(
            out_of_bounds_mask, 0
        )
        masked_w_adjacency_target = w_adjacency_target.masked_fill(
            out_of_bounds_mask, 0
        )
        is_correct = (
            th.sum(
                (masked_t_adjacency_guess != masked_t_adjacency_target)
                + (masked_w_adjacency_guess != masked_w_adjacency_target),
                dim=(-1, -2),
            )
            == 0
        )
        self.correct += th.sum(is_valid & is_correct).to(t_adjacency_guess.device)
        self.total += th.sum(is_valid).to(t_adjacency_guess.device)

    @override
    def compute(self) -> th.Tensor:
        """Compute the final true positive rate.

        Returns:
            The true positive rate as a tensor.
        """
        return self.correct.float() / self.total


class DiffusionWTPR(Metric):
    def __init__(
        self,
        njet_range: tuple[int, int] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._njet_range = njet_range
        self.add_state("correct", default=th.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=th.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        w_adjacency_guess: jx.Int[th.Tensor, "batch seq_len seq_len"],
        w_adjacency1: jx.Int[th.Tensor, " num_jets num_jets"],
        w_adjacency2: jx.Int[th.Tensor, " num_jets num_jets"],
        mask: jx.Bool[th.Tensor, "batch seq_len"],
        groups: list[jx.Int[th.Tensor, "n_groups"]],
    ):
        first_is_good = th.all(groups[0][:, 1:] >= 0, dim=-1)
        second_is_good = th.all(groups[1][:, 1:] >= 0, dim=-1)
        first_is_valid = compute_is_valid(first_is_good, mask, self._njet_range)
        second_is_valid = compute_is_valid(second_is_good, mask, self._njet_range)

        batch_size, max_seq_len, _ = w_adjacency_guess.shape
        seq_lens = th.sum(mask, dim=1)
        count_row = (
            th.arange(max_seq_len, device=mask.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, max_seq_len, 1)
        )
        count_col = (
            th.arange(max_seq_len, device=mask.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(batch_size, 1, max_seq_len)
        )
        out_of_bounds_mask = (
            (count_row >= seq_lens.unsqueeze(-1).unsqueeze(-1))
            | (count_col >= seq_lens.unsqueeze(-1).unsqueeze(-1))
        ).to(th.bool)
        masked_w_adjacency_guess = w_adjacency_guess.masked_fill(out_of_bounds_mask, 0)
        masked_w_adjacency1 = w_adjacency1.masked_fill(out_of_bounds_mask, 0)
        masked_w_adjacency2 = w_adjacency2.masked_fill(out_of_bounds_mask, 0)

        diagonal_mask = (
            th.eye(max_seq_len, device=mask.device)
            .bool()
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

        masked_w_adjacency1[diagonal_mask] = 0
        masked_w_adjacency2[diagonal_mask] = 0

        first_is_correct = (
            th.sum(
                masked_w_adjacency_guess < masked_w_adjacency1,
                dim=(-1, -2),
            )
            == 0
        )
        second_is_correct = (
            th.sum(
                masked_w_adjacency_guess < masked_w_adjacency2,
                dim=(-1, -2),
            )
            == 0
        )
        self.correct += th.sum(
            th.logical_and(first_is_correct, first_is_valid)
        ) + th.sum(th.logical_and(second_is_correct, second_is_valid))
        self.total += th.sum(first_is_valid) + th.sum(second_is_valid)

    @override
    def compute(self) -> th.Tensor:
        """Compute the final true positive rate.

        Returns:
            The true positive rate as a tensor.
        """
        return self.correct.float() / self.total


class DiffusionTopTPR(Metric):
    def __init__(
        self,
        njet_range: tuple[int, int] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._njet_range = njet_range
        self.add_state("correct", default=th.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=th.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        t_adjacency_guess: jx.Int[th.Tensor, "batch seq_len seq_len"],
        w_adjacency_guess: jx.Int[th.Tensor, "batch seq_len seq_len"],
        t_adjacency1: jx.Int[th.Tensor, " num_jets num_jets"],
        w_adjacency1: jx.Int[th.Tensor, " num_jets num_jets"],
        t_adjacency2: jx.Int[th.Tensor, " num_jets num_jets"],
        w_adjacency2: jx.Int[th.Tensor, " num_jets num_jets"],
        mask: jx.Bool[th.Tensor, "batch seq_len"],
        groups: list[jx.Int[th.Tensor, "n_groups"]],
    ):
        first_is_good = th.all(groups[0][:] >= 0, dim=-1)
        second_is_good = th.all(groups[1][:] >= 0, dim=-1)

        first_is_valid = compute_is_valid(first_is_good, mask, self._njet_range)
        second_is_valid = compute_is_valid(second_is_good, mask, self._njet_range)

        batch_size, max_seq_len, _ = t_adjacency_guess.shape
        seq_lens = th.sum(mask, dim=1)
        count_row = (
            th.arange(max_seq_len, device=mask.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, max_seq_len, 1)
        )
        count_col = (
            th.arange(max_seq_len, device=mask.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(batch_size, 1, max_seq_len)
        )
        out_of_bounds_mask = (
            (count_row >= seq_lens.unsqueeze(-1).unsqueeze(-1))
            | (count_col >= seq_lens.unsqueeze(-1).unsqueeze(-1))
        ).to(th.bool)
        masked_t_adjacency_guess = t_adjacency_guess.masked_fill(out_of_bounds_mask, 0)
        masked_w_adjacency_guess = w_adjacency_guess.masked_fill(out_of_bounds_mask, 0)
        masked_t_adjacency1 = t_adjacency1.masked_fill(out_of_bounds_mask, 0)
        masked_w_adjacency1 = w_adjacency1.masked_fill(out_of_bounds_mask, 0)
        masked_t_adjacency2 = t_adjacency2.masked_fill(out_of_bounds_mask, 0)
        masked_w_adjacency2 = w_adjacency2.masked_fill(out_of_bounds_mask, 0)

        diagonal_mask = (
            th.eye(max_seq_len, device=mask.device)
            .bool()
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        masked_t_adjacency1[diagonal_mask] = 0
        masked_w_adjacency1[diagonal_mask] = 0
        masked_t_adjacency2[diagonal_mask] = 0
        masked_w_adjacency2[diagonal_mask] = 0

        w1_is_correct = (
            th.sum(
                masked_w_adjacency_guess < masked_w_adjacency1,
                dim=(-1, -2),
            )
            == 0
        )
        t1_is_correct = (
            th.sum(
                masked_t_adjacency_guess < masked_t_adjacency1,
                dim=(-1, -2),
            )
            == 0
        )
        w2_is_correct = (
            th.sum(
                masked_w_adjacency_guess < masked_w_adjacency2,
                dim=(-1, -2),
            )
            == 0
        )
        t2_is_correct = (
            th.sum(
                masked_t_adjacency_guess < masked_t_adjacency2,
                dim=(-1, -2),
            )
            == 0
        )
        first_is_correct = th.logical_and(w1_is_correct, t1_is_correct)
        second_is_correct = th.logical_and(w2_is_correct, t2_is_correct)
        self.correct += th.sum(
            th.logical_and(first_is_correct, first_is_valid)
        ) + th.sum(th.logical_and(second_is_correct, second_is_valid))
        self.total += th.sum(first_is_valid) + th.sum(second_is_valid)

    @override
    def compute(self) -> th.Tensor:
        """Compute the final true positive rate.

        Returns:
            The true positive rate as a tensor.
        """
        return self.correct.float() / self.total
