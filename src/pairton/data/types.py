from typing import TypedDict

import jaxtyping as jx
import torch as th


class SequenceBatch(TypedDict):
    features: jx.Float[th.Tensor, " num_jets num_features"]
    mask: jx.Bool[th.Tensor, " num_jets"]
    groups: list[jx.Int[th.Tensor, " n_group"]]

    adjacency: jx.Int[th.Tensor, " num_jets num_jets"]  # Adjacency matrix (top)
    w_adjacency: jx.Int[th.Tensor, " num_jets num_jets"]

    mask_adjacency: jx.Bool[th.Tensor, " num_jets num_jets 2"]
    mask_level: jx.Int[th.Tensor, " 1"]

    w_adjacency1: jx.Int[th.Tensor, " num_jets num_jets"]
    w_adjacency2: jx.Int[th.Tensor, " num_jets num_jets"]

    top_adjacency1: jx.Int[th.Tensor, " num_jets num_jets"]
    top_adjacency2: jx.Int[th.Tensor, " num_jets num_jets"]

    feature_names: list[str]


class SequenceBatches(TypedDict):
    features: jx.Float[th.Tensor, "batch_size num_jets num_features"]
    mask: jx.Bool[th.Tensor, "batch_size num_jets"]
    groups: list[jx.Int[th.Tensor, "batch_size n_group"]]

    adjacency: jx.Int[th.Tensor, "batch_size num_jets num_jets"]  # Adjacency matrix
    w_adjacency: jx.Int[th.Tensor, "batch_size num_jets num_jets"]

    mask_adjacency: jx.Bool[th.Tensor, "batch_size num_jets num_jets 2"]
    mask_level: jx.Int[th.Tensor, " batch_size"]

    feature_names: list[str]
