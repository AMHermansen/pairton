from collections.abc import Callable
from dataclasses import dataclass
from itertools import chain, product

import jaxtyping as jx
import torch as th
from torch import nn



def maybe_expand_matrix_to_adjacency(
    matrix: jx.Float[th.Tensor, "batch seq_len seq_len ..."],
    adjacency: jx.Float[th.Tensor, "batch adj_seq_len adj_seq_len"],
) -> jx.Float[th.Tensor, "batch adj_seq_len adj_seq_len ..."]:
    """Expand matrix to match the size of the adjacency matrix if necessary.

    Args:
        matrix: Matrix to potentially expand.
        adjacency: Ground truth adjacency matrix.

    Returns:
        The matrix expanded to match the size of the adjacency matrix.
    """
    seq_len_diff = adjacency.shape[1] - matrix.shape[1]
    if seq_len_diff == 0:
        return matrix

    return th.nn.functional.pad(
        matrix,
        pad=(
            0,
            0,
            0,
            seq_len_diff,
            0,
            seq_len_diff,
        ),
        mode="constant",
        value=0,
    )


def square_inverse_weight_fn(
    visible_tokens: jx.Int[th.Tensor, " batch"],
) -> jx.Float[th.Tensor, " batch"]:
    """Compute weights inversely proportional to the square of the number of visible tokens.

    Args:
        visible_tokens: Number of visible tokens in each item of the batch.

    Returns:
        Weights for each item in the batch.
    """
    return 1.0 / (visible_tokens.float() ** 2 + 1e-8)


def compute_losses_from_adjancency_matrix(
    adjacency: jx.Float[th.Tensor, "batch seq_len seq_len"],
    logits: jx.Float[th.Tensor, "batch seq_len seq_len 1"],
    mask: jx.Bool[th.Tensor, "batch seq_len"],
    visible_token_weight_fn: Callable[
        [jx.Int[th.Tensor, " batch"]], jx.Float[th.Tensor, " batch"]
    ] = square_inverse_weight_fn,
) -> tuple[jx.Float[th.Tensor, "1"], jx.Float[th.Tensor, " batch"]]:
    """Compute binary cross-entropy loss from adjacency matrix predictions.

    Args:
        adjacency: Ground truth adjacency matrix.
        logits: Predicted logits for the adjacency matrix.
        mask: Mask indicating valid tokens.
        visible_token_weight_fn: Function that computes weights for visible tokens.

    Returns:
        A tuple containing:
            - The mean loss across the batch.
            - The individual losses for each item in the batch.
    """
    visible_tokens = mask.sum(dim=-1)
    max_visible = mask.shape[1]
    adjacency = adjacency[:, :max_visible, :max_visible]
    logits = maybe_expand_matrix_to_adjacency(logits, adjacency)
    full_losses = nn.functional.binary_cross_entropy_with_logits(
        input=logits.squeeze(-1),
        target=adjacency.float(),
        reduction="none",
    )
    mask_matrix = mask[:, :, None] * mask[:, None, :]
    losses = th.einsum(
        "b i j, b i j -> b", full_losses, mask_matrix.to(th.float32)
    ) * visible_token_weight_fn(visible_tokens)
    loss = losses.mean()
    return loss, losses


def construct_mask_matrix(
    mask: jx.Bool[th.Tensor, "batch seq_len"],
) -> jx.Bool[th.Tensor, "batch seq_len seq_len"]:
    """Construct a mask matrix from a 1D mask.

    Args:
        mask: 1D mask indicating valid tokens.

    Returns:
        2D mask matrix.
    """
    return mask[:, :, None] & mask[:, None, :]


def compute_pair_features(
    single_features: jx.Float[th.Tensor, "batch seq_len 6"],
    eps=1e-4,
) -> jx.Float[th.Tensor, "batch seq_len seq_len 6"]:
    """Compute pairwise features from single features.

    Args:
        single_features: Single features of shape (batch, seq_len, 6). These are
            expected to be
            - log(E) - 4.5
            - log(PT) - 4.5
            - eta
            - cos(phi)
            - sin(phi)
            - is_tagged.
            - ...
            If this is not the case, the returned values may not be meaningful!
        eps: Small value to avoid numerical issues.

    Returns:
        Pairwise features of shape (batch, seq_len, seq_len, 6).
    """
    # Constant
    # This might become a parameter later. For now, we just hardcode it, since it would
    # require a refactoring of the data pipeline to make it configurable, and
    # incorporate this function.
    log_offset_constant = 4.5

    d_eta = th.cdist(single_features[..., 2:3], single_features[..., 2:3], p=1)
    d_phi_cos = th.cdist(single_features[..., 3:4], single_features[..., 3:4], p=1)
    d_phi_sin = th.cdist(single_features[..., 4:5], single_features[..., 4:5], p=1)
    d_xy_normalized = th.stack([d_phi_cos, d_phi_sin], dim=-1) * (
        (d_phi_sin**2 + d_phi_cos**2).clamp(min=eps) ** (-0.5)
    ).unsqueeze(-1)

    pT_normal_units = th.exp(single_features[..., 1] + log_offset_constant)
    energy_normal_units = th.exp(single_features[..., 0] + log_offset_constant)
    px = pT_normal_units * single_features[..., 3]
    py = pT_normal_units * single_features[..., 4]
    pz = pT_normal_units * th.sinh(single_features[..., 2])

    mass = (
        th.clamp(
            th.cdist(
                energy_normal_units.unsqueeze(-1),
                -energy_normal_units.unsqueeze(-1),
                p=2,
            )
            ** 2
            - (
                th.cdist(px.unsqueeze(-1), -px.unsqueeze(-1), p=2) ** 2
                + th.cdist(py.unsqueeze(-1), -py.unsqueeze(-1), p=2) ** 2
                + th.cdist(pz.unsqueeze(-1), -pz.unsqueeze(-1), p=2) ** 2
            ),
            min=eps,
        )
        ** 0.5
    )
    mass_standardized = th.log(mass + eps) - log_offset_constant
    phi = th.atan2(single_features[..., 4], single_features[..., 3])
    d_phi = phi.unsqueeze(-1) - phi.unsqueeze(-2)
    d_phi = (d_phi + th.pi) % (2 * th.pi) - th.pi
    d_r = (d_eta**2 + d_phi**2 + eps).sqrt()
    return th.cat(
        [
            d_eta.unsqueeze(-1),
            d_phi.unsqueeze(-1),
            d_r.unsqueeze(-1),
            mass_standardized.unsqueeze(-1),
            d_xy_normalized,
        ],
        dim=-1,
    )


@dataclass
class OptimizeConfig:
    compile_components: bool = False
    remove_redundant_padding: bool = True


def _mask_diagonal(
    matrix: jx.Float[th.Tensor, "seq_len seq_len"],
) -> jx.Float[th.Tensor, "seq_len seq_len"]:
    """Mask the diagonal of a matrix with negative infinity."""
    maxtrix_copy = matrix.clone()
    diag = th.eye(matrix.shape[0], device=matrix.device, dtype=th.bool)
    maxtrix_copy[diag] = -th.inf
    return maxtrix_copy


def compute_gain_from_pair_logits(
    logit_matrix: jx.Float[th.Tensor, "seq_len seq_len"],
    edge_weight: float = 2.0,
) -> jx.Float[th.Tensor, "seq_len seq_len"]:
    """Calculate pairwise scores by subtracting diagonal elements."""
    diagonal = th.diagonal(logit_matrix)
    return edge_weight * logit_matrix - diagonal.unsqueeze(0) - diagonal.unsqueeze(1)


def _find_max_indices(
    matrix: jx.Float[th.Tensor, "seq_len seq_len"],
) -> tuple[th.Tensor, th.Tensor]:
    """Find the indices of the maximum value in a 2D matrix."""
    indices = th.unravel_index(matrix.argmax(), matrix.shape)
    assert len(indices) == 2
    return indices


def _compute_gain_for_edges(
    logits: jx.Float[th.Tensor, "seq_len seq_len"],
    edges: tuple[jx.Int[th.Tensor, "1"], jx.Int[th.Tensor, "1"]],
) -> jx.Float[th.Tensor, "seq_len"]:
    """Compute the gain for connecting edges to each node."""
    edge1_gain = logits[edges[0], :] - th.diagonal(logits)
    edge2_gain = logits[edges[1], :] - th.diagonal(logits)
    return edge1_gain + edge2_gain


def _select_best_b_nodes(
    top_logits: jx.Float[th.Tensor, "seq_len seq_len"],
    edges1: tuple[jx.Int[th.Tensor, "1"], jx.Int[th.Tensor, "1"]],
    edges2: tuple[jx.Int[th.Tensor, "1"], jx.Int[th.Tensor, "1"]],
) -> tuple[jx.Int[th.Tensor, "1"], jx.Int[th.Tensor, "1"]]:
    """Select the best two B nodes based on gain calculation."""
    logits_copy = top_logits.clone()

    # Compute gains for both edge pairs (same as original)
    gain_from_edges1 = (
        logits_copy[edges1[0], :] + logits_copy[edges1[1], :] - th.diagonal(logits_copy)
    ).squeeze(0)
    gain_from_edges2 = (
        logits_copy[edges2[0], :] + logits_copy[edges2[1], :] - th.diagonal(logits_copy)
    ).squeeze(0)

    # Mask out already chosen nodes.
    all_edge_nodes = list(chain(edges1, edges2))
    for gain_tensor in [gain_from_edges1, gain_from_edges2]:
        for node_idx in all_edge_nodes:
            gain_tensor[node_idx] = -th.inf

    # Find the pair that maximizes total gain (same as original)
    total_gain = gain_from_edges1.unsqueeze(1) + gain_from_edges2.unsqueeze(0)
    total_gain = _mask_diagonal(total_gain)
    best_node_indices = _find_max_indices(total_gain)

    return best_node_indices


def _mask_nodes_in_matrix(
    matrix: jx.Float[th.Tensor, "seq_len seq_len"],
    node_indices: tuple[jx.Int[th.Tensor, "1"], jx.Int[th.Tensor, "1"]],
) -> jx.Float[th.Tensor, "seq_len seq_len"]:
    """Mask out rows and columns corresponding to given node indices."""
    masked_matrix = matrix.clone()
    for node_idx in node_indices:
        masked_matrix[node_idx, :] = -th.inf
        masked_matrix[:, node_idx] = -th.inf
    return masked_matrix


def _toggle_edges_in_adjacency(
    adjacency: jx.Int[th.Tensor, "seq_len seq_len"],
    edge_indices: tuple[th.Tensor, th.Tensor],
) -> None:
    """Toggle edges in the adjacency matrix for given indices."""
    for idx1, idx2 in product(edge_indices, repeat=2):
        adjacency[idx1, idx2] ^= 1  # Toggle edge


def _create_initial_adjacency(
    seq_len: int, device: th.device
) -> jx.Int[th.Tensor, "seq_len seq_len"]:
    """Create initial adjacency matrix as identity matrix."""
    return th.eye(seq_len, seq_len, device=device, dtype=th.int32)


def create_fast_inference_guess(
    logits: jx.Float[th.Tensor, "seq_len seq_len 2"],
    use_mixed_logits_for_w: bool = True,
) -> tuple[jx.Int[th.Tensor, "seq_len seq_len"], jx.Int[th.Tensor, "seq_len seq_len"]]:
    """Create a fast inference guess from logits.

    Args:
        logits: Logits for the adjacency matrix with shape (seq_len, seq_len, 2).
        use_mixed_logits_for_w: Whether to use mixed logits for W structure scoring.

    Returns:
        A tuple containing:
        - w_adjacency: Binary adjacency matrix for W structure
        - top_adjacency: Binary adjacency matrix for TOP structure

    Raises:
        ValueError: If logits are not on CPU device.
    """
    # IMPORTANT: We raise error, because performance is ~5x worse on GPU for this function.
    # (Performance run on batch size 256): CPU: 0.1986s vs GPU: 0.9596s
    if logits.device != th.device("cpu"):
        raise ValueError("Logits must be on CPU device for fast inference guess.")

    top_logits, w_logits = logits[..., 0], logits[..., 1]
    seq_len, device = logits.shape[0], logits.device

    w_scores = compute_gain_from_pair_logits(w_logits)

    if use_mixed_logits_for_w:
        top_scores = compute_gain_from_pair_logits(top_logits)
        combined_scores = top_scores + w_scores
    else:
        combined_scores = w_scores

    combined_scores = _mask_diagonal(combined_scores)

    best_edges = _find_max_indices(combined_scores)
    combined_scores = _mask_nodes_in_matrix(combined_scores, best_edges)
    second_best_edges = _find_max_indices(combined_scores)

    w_adjacency = _create_initial_adjacency(seq_len, device)
    _toggle_edges_in_adjacency(w_adjacency, best_edges)
    _toggle_edges_in_adjacency(w_adjacency, second_best_edges)
    top_adjacency = w_adjacency.clone()

    best_b_nodes = _select_best_b_nodes(top_logits, best_edges, second_best_edges)

    for b_node in best_b_nodes:
        top_adjacency[b_node, b_node] ^= 1

    # Connect first set of edges to first B node
    for edge_node in best_edges:
        top_adjacency[edge_node, best_b_nodes[0]] ^= 1
        top_adjacency[best_b_nodes[0], edge_node] ^= 1

    # Connect second set of edges to second B node
    for edge_node in second_best_edges:
        top_adjacency[edge_node, best_b_nodes[1]] ^= 1
        top_adjacency[best_b_nodes[1], edge_node] ^= 1

    return w_adjacency, top_adjacency
