import warnings
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import override

import torch as th
from einops import rearrange
from lightning import LightningModule
from torch import nn

from pairton.data.types import SequenceBatch
from pairton.layers.symmetric import MultiDimensionalLinearNoBiasThenOuterSum
from pairton.models.utils import (
    compute_gain_from_pair_logits,
)
from pairton.models.metrics import (
    DiffusionFullTPR,
    DiffusionTopTPR,
    DiffusionWTPR,
)
from pairton.models.utils import compute_pair_features, construct_mask_matrix
from ml_utils.components import MLP, SwiGLUMLP
from ml_utils.lightning_utils import LightningConfig, configure_optimizer_standard
from ml_utils.torch_utils.sinkhorn_knopp import sinkhorn_knopp
from ml_utils.utils import default, exists

from .base import SequenceWrapperModel


def symmetrize_pair_output(pair_output: th.Tensor) -> th.Tensor:
    """Symmetrize pair output by averaging with its transpose.

    Args:
        pair_output: Tensor of shape (batch_size, seq_len, seq_len, dim)

    Returns:
        symmetrized_pair_output: Tensor of shape (batch_size, seq_len, seq_len, dim)
    """
    return (pair_output + rearrange(pair_output, "b i j d -> b j i d")) / 2


@dataclass
class DiffusionConfig:
    """Configuration for diffusion models."""

    use_discrete_diffusion: bool = False
    output_hypergraph: bool = False
    predict_next_step_only: bool = False
    use_split_b_prediction: bool = False

    def __post_init__(self):
        if not self.use_discrete_diffusion:
            if self.output_hypergraph:
                warnings.warn(
                    "Outputting hypergraph without discrete diffusion is not possible."
                    "Output will be changed to adjacency matrices.",
                    UserWarning,
                )
                self.output_hypergraph = False
            if self.predict_next_step_only:
                warnings.warn(
                    "Predicting next step only without discrete diffusion is not possible."
                    "Setting predict_next_step_only to False.",
                    UserWarning,
                )
                self.predict_next_step_only = False


@dataclass
class TopIndexCollection:
    """Collection of indices for top-quark reconstruction.

    Attributes:
        idx_b: Indices of predicted b-quarks.
        idx_w1: Indices of first W-boson quark.
        idx_w2: Indices of second W-boson quark.
    """

    idx_b: th.Tensor
    idx_w1: th.Tensor
    idx_w2: th.Tensor


class Pairton(LightningModule):
    """A general sequence model for combinatorial sequences."""

    def __init__(
        self,
        in_features: int,
        sequence_model: SequenceWrapperModel,
        *,
        lightning_config: LightningConfig | None = None,
        sinkhorn_knopp_iterations: int = 0,
        diffusion_config: DiffusionConfig | None = None,
        remove_redundant_padding: bool = True,
        include_dxy: bool = True,
        use_pair_output: bool = False,
    ):
        super().__init__()
        diffusion_config = default(diffusion_config, DiffusionConfig())
        self.save_hyperparameters(
            ignore=[
                "sequence_model",
            ]
        )

        self._diffusion_config = diffusion_config
        self._use_discrete_diffusion = diffusion_config.use_discrete_diffusion
        self._output_hypergraph = diffusion_config.output_hypergraph
        self._predict_next_step_only = diffusion_config.predict_next_step_only
        self._use_split_b_prediction = diffusion_config.use_split_b_prediction

        self._in_features = in_features
        self._sequence_model = sequence_model
        self._lightning_config = default(lightning_config, LightningConfig())
        self._sinkhorn_knopp_iterations = sinkhorn_knopp_iterations
        self._remove_redundant_padding = remove_redundant_padding
        self._include_dxy = include_dxy
        self._use_pair_output = use_pair_output

        self._setup_model()

    @property
    def include_pair_features(self) -> bool:
        """Whether the model includes pair features."""
        return exists(self._sequence_model.dim_pair)

    # LightningModule methods
    @override
    def forward(
        self, batch: SequenceBatch
    ) -> tuple[th.Tensor, th.Tensor | None, th.Tensor]:
        if self._remove_redundant_padding:
            batch = self._remove_padding(batch)
        single_representations, pair_representations, mask = self._get_embeddings(batch)
        (
            embedded_sequences,
            embedded_pair_sequences,
            mask,
        ) = self._sequence_model(
            single_features=single_representations,
            pair_features=pair_representations,
            mask=mask,
        )
        return (
            self._post_model_sequence_norm(embedded_sequences),
            self._post_model_pair_norm(embedded_pair_sequences),
            mask,
        )

    @override
    def training_step(
        self,
        batch: SequenceBatch,
        batch_idx: int,
    ):
        del batch_idx  # Unused, match LightningModule signature
        latent_sequences, latent_pairs, mask = self(batch)
        loss = self._compute_loss(
            latent_sequences=latent_sequences,
            latent_pairs=latent_pairs,
            mask=mask,
            batch=batch,
        )
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return {
            "loss": loss,
            "latent_sequences": latent_sequences,
        }

    @override
    def validation_step(
        self,
        batch: SequenceBatch,
        batch_idx: int,
    ):
        mask = batch["mask"]
        (
            t_adjacency_predict,
            w_adjacency_predict,
            t_adjacency_truth,
            w_adjacency_truth,
            top1_idx,
            top2_idx,
        ) = self.predict_adjacency(batch)

        self._compute_metrics(
            batch,
            mask,
            t_adjacency_predict,
            t_adjacency_truth,
            w_adjacency_predict,
            w_adjacency_truth,
        )
        self._log_metrics(prefix="val")
        return {
            "t_adjacency_predict": t_adjacency_predict,
            "w_adjacency_predict": w_adjacency_predict,
            "t_adjacency_target": t_adjacency_truth,
            "w_adjacency_target": w_adjacency_truth,
            "top1_idx": top1_idx,
            "top2_idx": top2_idx,
        }

    @override
    def test_step(self, batch: SequenceBatch, batch_idx: int):
        mask = batch["mask"]
        (
            t_adjacency_predict,
            w_adjacency_predict,
            t_adjacency_truth,
            w_adjacency_truth,
            top1_idx,
            top2_idx,
        ) = self.predict_adjacency(batch)

        self._compute_metrics(
            batch,
            mask,
            t_adjacency_predict,
            t_adjacency_truth,
            w_adjacency_predict,
            w_adjacency_truth,
        )
        self._log_metrics(prefix="test")
        return {
            "t_adjacency_predict": t_adjacency_predict,
            "w_adjacency_predict": w_adjacency_predict,
            "t_adjacency_target": t_adjacency_truth,
            "w_adjacency_target": w_adjacency_truth,
            "top1_idx": top1_idx,
            "top2_idx": top2_idx,
        }

    @override
    def predict_step(self, batch: SequenceBatch, batch_idx: int):
        (
            t_adjacency_predict,
            w_adjacency_predict,
            t_adjacency_truth,
            w_adjacency_truth,
            top1_idx,
            top2_idx,
        ) = self.predict_adjacency(batch)
        groups = batch["groups"]
        return {
            "t_adjacency_predict": t_adjacency_predict,
            "w_adjacency_predict": w_adjacency_predict,
            "t_adjacency_target": t_adjacency_truth,
            "w_adjacency_target": w_adjacency_truth,
            "top1_idx": top1_idx,
            "top2_idx": top2_idx,
            "groups": groups,
        }

    @override
    def configure_optimizers(self):
        """Configure the optimizers and learning rate schedulers.

        Returns: A dictionary containing the optimizer and scheduler.
        """
        return configure_optimizer_standard(
            self,
            self._lightning_config,
        )

    # Metric computation and logging
    def _compute_metrics(
        self,
        batch: SequenceBatch,
        mask: th.Tensor,
        t_adjacency_predict: th.Tensor,
        t_adjacency_truth: th.Tensor,
        w_adjacency_predict: th.Tensor,
        w_adjacency_truth: th.Tensor,
    ):
        for metric in self._val_w_diffusion_metrics.values():
            metric(
                w_adjacency_predict,
                batch["w_adjacency1"],
                batch["w_adjacency2"],
                mask,
                batch["groups"],
            )
        for metric in self._val_t_diffusion_metrics.values():
            metric(
                t_adjacency_predict,
                w_adjacency_predict,
                batch["top_adjacency1"],
                batch["w_adjacency1"],
                batch["top_adjacency2"],
                batch["w_adjacency2"],
                mask,
                batch["groups"],
            )
        for metric in self._val_event_diffusion_metrics.values():
            metric(
                t_adjacency_predict,
                w_adjacency_predict,
                t_adjacency_truth,
                w_adjacency_truth,
                mask,
                batch["groups"],
            )
        self._val_w_diffusion(
            w_adjacency_predict,
            batch["w_adjacency1"],
            batch["w_adjacency2"],
            mask,
            batch["groups"],
        )
        self._val_t_diffusion(
            t_adjacency_predict,
            w_adjacency_predict,
            batch["top_adjacency1"],
            batch["w_adjacency1"],
            batch["top_adjacency2"],
            batch["w_adjacency2"],
            mask,
            batch["groups"],
        )
        self._val_event_full_diffusion(
            t_adjacency_predict,
            w_adjacency_predict,
            t_adjacency_truth,
            w_adjacency_truth,
            mask,
            batch["groups"],
        )

    def _log_metrics(self, prefix: str = "val"):
        # Old metrics. Kept for reference.
        self.log(
            f"{prefix}/t_diffusion_tpr",
            self._val_t_diffusion,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{prefix}/w_diffusion_tpr",
            self._val_w_diffusion,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{prefix}/event_full_diffusion_tpr",
            self._val_event_full_diffusion,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        for name, metric in self._val_t_diffusion_metrics.items():
            self.log(
                f"{prefix}/t_diffusion_tpr_{name}",
                metric,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
        for name, metric in self._val_w_diffusion_metrics.items():
            self.log(
                f"{prefix}/w_diffusion_tpr_{name}",
                metric,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
        for name, metric in self._val_event_diffusion_metrics.items():
            self.log(
                f"{prefix}/event_full_diffusion_tpr_{name}",
                metric,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )

    # Setup methods
    def _setup_model(self):
        # Determine output features
        # If outputting hypergraph for top cluster, we have 1 adjacency matrix (W-boson)
        # and 1 3d tensor containing hyperedges.
        # Otherwise, we have two adjacency matrices (t-quark and W-boson).
        pair_output_features = 1 if self._output_hypergraph else 2
        out_3d_features = 1 if self._output_hypergraph else None

        self._setup_embeddings()
        self._setup_essential_metrics()
        self._setup_output_functions(pair_output_features, out_3d_features)

    def _setup_embeddings(self):
        self._single_embed = nn.Sequential(
            nn.Linear(
                in_features=self._in_features,
                out_features=self._sequence_model.dim_single,
                bias=False,
            ),
            nn.SiLU(),
            SwiGLUMLP(in_features=self._sequence_model.dim_single),
        )
        self._post_model_sequence_norm = nn.RMSNorm(self._sequence_model.dim_single)
        if self._use_pair_output:
            self._post_model_pair_norm = nn.RMSNorm(self._sequence_model.dim_pair)
        else:
            self._post_model_pair_norm = nn.Identity()
        if self.include_pair_features:
            assert isinstance(self._sequence_model.dim_pair, int)
            self._sequence_to_pair_embed = nn.Sequential(
                MultiDimensionalLinearNoBiasThenOuterSum(
                    dim=self._in_features,
                    repeats=[1, 1],
                    dim_out=self._sequence_model.dim_pair,
                ),
                nn.LayerNorm(self._sequence_model.dim_pair),
                nn.SiLU(),
                SwiGLUMLP(in_features=self._sequence_model.dim_pair),
            )
            # Currently hardcoding pair input features, due to lack of config in pipeline
            input_pair_features = 6 if self._include_dxy else 4
            self._pair_to_pair_embed = MLP(
                in_features=input_pair_features,
                out_features=self._sequence_model.dim_pair,
            )
            if self._use_discrete_diffusion:
                # Adjacency embeddings for discrete diffusion
                # The 3 embeddings correspond to: no edge, edge, unknown.
                #
                self._t_adjacency_embedding = nn.Embedding(
                    num_embeddings=3, embedding_dim=self._sequence_model.dim_pair // 2
                )
                self._w_adjacency_embedding = nn.Embedding(
                    num_embeddings=3, embedding_dim=self._sequence_model.dim_pair // 2
                )
        else:
            if self._use_discrete_diffusion:
                raise ValueError(
                    "Discrete diffusion requires pair embeddings in the sequence model."
                )
            self._sequence_to_pair_embed = None
            self._pair_to_pair_embed = None

    def _setup_essential_metrics(self):
        name_njet_ranges = {
            "njet_6-6": (6, 6),
            "njet_7-7": (7, 7),
            "njet_8-20": (8, 20),
            "njet_6-20": (6, 20),  # Full range
        }

        self._val_event_diffusion_metrics = nn.ModuleDict({
            name: DiffusionFullTPR(njet_range=njet_range)
            for name, njet_range in name_njet_ranges.items()
        })
        self._val_w_diffusion_metrics = nn.ModuleDict({
            name: DiffusionWTPR(njet_range=njet_range)
            for name, njet_range in name_njet_ranges.items()
        })
        self._val_t_diffusion_metrics = nn.ModuleDict({
            name: DiffusionTopTPR(njet_range=njet_range)
            for name, njet_range in name_njet_ranges.items()
        })
        self._val_event_full_diffusion = DiffusionFullTPR()
        self._val_w_diffusion = DiffusionWTPR()
        self._val_t_diffusion = DiffusionTopTPR()

    def _setup_output_functions(
        self,
        pair_output_features: int,
        out_3d_features: int | None,
    ):
        self._sequence_to_pair_output_embed = MultiDimensionalLinearNoBiasThenOuterSum(
            self._sequence_model.dim_single,
            repeats=[2],
        )
        self._pair_to_output = nn.Sequential(
            nn.LayerNorm(self._sequence_model.dim_single),
            nn.SiLU(),
            nn.Linear(
                in_features=self._sequence_model.dim_single,
                out_features=pair_output_features,
            ),
        )
        if self._use_pair_output:
            if not exists(self._sequence_model.dim_pair):
                raise ValueError(
                    "Pair output requires pair features in the sequence model."
                )
            self._pair_to_pair_output_embed = nn.Linear(
                in_features=self._sequence_model.dim_pair,
                out_features=self._sequence_model.dim_single,  # We upscale dimension to match single embed
            )

        if exists(out_3d_features):
            self._output_3d_func = nn.Sequential(
                MultiDimensionalLinearNoBiasThenOuterSum(
                    self._sequence_model.dim_single,
                    repeats=[2, 1],
                ),
                nn.LayerNorm(self._sequence_model.dim_single),
                nn.SiLU(),
                nn.Linear(self._sequence_model.dim_single, out_3d_features),
            )

    # Internal inference methods
    def _get_embeddings(
        self, batch: SequenceBatch
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        single_representations = self._single_embed(batch["features"])
        mask = batch["mask"]

        if self._use_discrete_diffusion:
            adjacency_mask = batch["mask_adjacency"]
            visible_t_adj = batch["adjacency"].clone()
            visible_w_adj = batch["w_adjacency"].clone()
            visible_t_adj[~adjacency_mask[..., 0]] = 2  # Mark as unknown
            visible_w_adj[~adjacency_mask[..., 1]] = 2  # Mark as unknown

            t_adj_embedding = self._t_adjacency_embedding(visible_t_adj)
            w_adj_embedding = self._w_adjacency_embedding(visible_w_adj)

            adj_embedding = th.cat([t_adj_embedding, w_adj_embedding], dim=-1)

        else:
            adj_embedding = 0

        if self.include_pair_features:
            pairwise_from_single_representations = self._sequence_to_pair_embed(
                batch["features"]
            )
            pairwise_from_pair_features = self._pair_to_pair_embed(
                self._construct_pair_features(batch["features"])
            )
            total_pairwise_representations = (
                pairwise_from_single_representations
                + pairwise_from_pair_features
                + adj_embedding
            )
        else:
            total_pairwise_representations = adj_embedding

        return single_representations, total_pairwise_representations, mask

    def _output_pair_func(
        self,
        latent_sequences: th.Tensor,
        latent_pairwise: th.Tensor | None = None,
    ):
        if self._use_pair_output:
            if not exists(latent_pairwise):
                raise ValueError(
                    "Pair output requires pair features from the sequence model."
                )
            pair_embedded = self._pair_to_pair_output_embed(
                symmetrize_pair_output(latent_pairwise)
            )
        else:
            pair_embedded = 0
        single_embeded_pair = self._sequence_to_pair_output_embed(latent_sequences)
        combined_pair_features = single_embeded_pair + pair_embedded
        return self._pair_to_output(combined_pair_features)

    def _construct_pair_features(self, features: th.Tensor) -> th.Tensor:
        full_pair_features = compute_pair_features(features)
        if self._include_dxy:
            return full_pair_features
        return full_pair_features[..., :4]

    def _remove_padding(self, batch: SequenceBatch) -> SequenceBatch:
        seq_lens = batch["mask"].sum(dim=1)
        max_seq_len = seq_lens.max().item()
        if max_seq_len < batch["mask"].shape[-1]:
            batch["features"] = batch["features"][:, :max_seq_len, :]
            batch["mask"] = batch["mask"][:, :max_seq_len]
            if "adjacency" in batch:
                batch["adjacency"] = batch["adjacency"][
                :, :max_seq_len, :max_seq_len, :
                ]
            if "mask_adjacency" in batch:
                batch["mask_adjacency"] = batch["mask_adjacency"][
                :, :max_seq_len, :max_seq_len, :
                ]
        return batch

    def _compute_loss(
        self,
        latent_sequences: th.Tensor,
        latent_pairs: th.Tensor | None,
        mask: th.Tensor,
        batch: SequenceBatch,
    ):
        """Compute loss.

        Args:
            latent_sequences: Tensor of shape (batch_size, seq_len, dim)
            latent_pairs: Tensor of shape (batch_size, seq_len, seq_len, dim) or None
            mask: Tensor of shape (batch_size, seq_len)
            batch: Input batch.

        Returns:
            loss: Computed loss.
        """
        pair_output = self._output_pair_func(latent_sequences, latent_pairs)
        w_losses = self._compute_w_loss(pair_output, mask, batch)
        if self._output_hypergraph:
            output_hyperedge_3d = self._output_3d_func(latent_sequences)
            top_losses = self._compute_top_3d_loss(output_hyperedge_3d, mask, batch)
        else:
            top_losses = self._compute_top_loss(pair_output, mask, batch)
        return th.cat([w_losses, top_losses], dim=0).mean()

    def _compute_w_loss(
        self,
        pair_output: th.Tensor,
        mask: th.Tensor,
        batch: SequenceBatch,
    ) -> th.Tensor:
        """Compute W-boson adjacency loss.

        Args:
            pair_output: Tensor of shape (batch_size, seq_len, seq_len, dim)
            mask: Tensor of shape (batch_size, seq_len)
            batch: Input batch.

        Returns:
            loss: Computed W-boson reconstruction loss.
        """
        w_output_matrix = pair_output[..., -1]
        mask_levels = batch["mask_level"]
        predict_w_adjacency = mask_levels < 2

        w_target = batch["w_adjacency"]
        adjacency_mask = batch["mask_adjacency"]

        w_mask = (
            ~adjacency_mask[..., -1] & predict_w_adjacency.unsqueeze(-1).unsqueeze(-1)
            if self._diffusion_config.predict_next_step_only
            else ~adjacency_mask[..., -1]
        )
        w_target_masked = w_target[w_mask]

        if self._sinkhorn_knopp_iterations:
            mask_matrix = construct_mask_matrix(mask).to(w_output_matrix.dtype)
            pre_probababilities = th.exp(w_output_matrix * mask_matrix)
            w_probabilities = sinkhorn_knopp(
                pre_probababilities,
                dim1=1,
                dim2=2,
                num_iters=self._sinkhorn_knopp_iterations,
            )
            w_probabilities_masked = w_probabilities[w_mask]
            with th.cuda.amp.autocast(enabled=False):
                w_losses = nn.functional.binary_cross_entropy(
                    w_probabilities_masked.float(),
                    w_target_masked.float(),
                    reduction="none",
                )
        else:
            w_output_masked = w_output_matrix[w_mask]
            w_losses = nn.functional.binary_cross_entropy_with_logits(
                w_output_masked.float(),
                w_target_masked.float(),
                reduction="none",
            )
        return w_losses

    def _compute_top_loss(
        self,
        pair_output: th.Tensor,
        mask: th.Tensor,
        batch: SequenceBatch,
    ) -> th.Tensor:
        """Compute top-quark adjacency loss.

        Args:
            pair_output: Tensor of shape (batch_size, seq_len, seq_len, dim)
            mask: Tensor of shape (batch_size, seq_len)
            batch: Input batch.

        Returns:
            loss: Computed top-quark reconstruction loss.
        """
        del mask  # Unused
        top_output_matrix = pair_output[..., 0]

        mask_levels = batch["mask_level"]

        predict_top_adjacency = mask_levels >= 2

        top_target = batch["adjacency"]
        adjacency_mask = batch["mask_adjacency"]

        top_mask = (
            ~adjacency_mask[..., 0] & predict_top_adjacency.unsqueeze(-1).unsqueeze(-1)
            if self._diffusion_config.predict_next_step_only
            else ~adjacency_mask[..., 0]
        )
        top_target_masked = top_target[top_mask]

        top_output_masked = top_output_matrix[top_mask]
        return nn.functional.binary_cross_entropy_with_logits(
            top_output_masked.float(),
            top_target_masked.float(),
            reduction="none",
        )

    def _compute_top_3d_loss(
        self,
        adjacency_3d_output: th.Tensor,
        mask: th.Tensor,
        batch: SequenceBatch,
    ) -> th.Tensor:
        """Compute top-quark hyperedge loss.

        Args:
            adjacency_3d_output: Tensor of shape (batch_size, seq_len, seq_len, seq_len, dim)
            mask: Tensor of shape (batch_size, seq_len)
            batch: Input batch.

        Returns:
            loss: Computed top-quark hyperedge reconstruction loss.
        """
        group1, group2 = batch["groups"]
        group1_predictions = self._get_3d_group_predictions(adjacency_3d_output, group1)
        group2_predictions = self._get_3d_group_predictions(adjacency_3d_output, group2)
        group1_losses = nn.functional.cross_entropy(
            group1_predictions,
            group1[:, 0].to(th.long),
            reduction="none",
        )
        group2_losses = nn.functional.cross_entropy(
            group2_predictions,
            group2[:, 0].to(th.long),
            reduction="none",
        )
        return (group1_losses.mean() + group2_losses.mean()) / 2

    def _get_3d_group_predictions(
        self,
        adjacency_3d_output: th.Tensor,
        group: th.Tensor,
    ):
        """Compute group predictions from 3D adjacency output.

        Args:
            adjacency_3d_output: Hyperedge adjacency output tensor.
            group: Group tensor indicating particle groupings.

        Returns:
            Logits for the missing b-quark in each triplet. Assumes W-boson is given.
        """
        logits_3d = adjacency_3d_output.squeeze(-1)
        batch_idx = th.arange(len(logits_3d), device=logits_3d.device, dtype=th.long)
        return logits_3d[
            batch_idx,
            group[:, 1],
            group[:, 2],
        ]

    # Prediction inference methods
    @th.no_grad()
    def predict_adjacency(
        self,
        batch: SequenceBatch,
    ) -> tuple[
        th.Tensor,
        th.Tensor,
        th.Tensor,
        th.Tensor,
        TopIndexCollection,
        TopIndexCollection,
    ]:
        """Predict adjacency matrices for top-quark and W-boson.

        Args:
            batch: Input batch.

        Returns:
            top_adjacency: Predicted top-quark adjacency matrix.
            w_adjacency: Predicted W-boson adjacency matrix.
            truth_top_adjacency: Ground truth top-quark adjacency matrix.
            truth_w_adjacency: Ground truth W-boson adjacency matrix.
            top1_idx: Indices of first predicted top-quark.
            top2_idx: Indices of second predicted top-quark.
        """
        batch_size, seq_len = batch["mask"].shape

        no_info_batch = deepcopy(batch)
        no_info_batch["mask_adjacency"] = th.zeros(
            (batch_size, seq_len, seq_len, 2),
            dtype=th.bool,
            device=batch["mask"].device,
        )
        target_t_adjacency = batch["adjacency"]
        target_w_adjacency = batch["w_adjacency"]
        no_info_batch["adjacency"] = th.zeros(
            (batch_size, seq_len, seq_len),
            dtype=th.long,
            device=batch["mask"].device,
        )
        no_info_batch["w_adjacency"] = th.zeros(
            (batch_size, seq_len, seq_len),
            dtype=th.long,
            device=batch["mask"].device,
        )
        # Creating indices for efficient slicing later
        diagonal_indices = th.arange(seq_len, device=batch["mask"].device)
        batch_indices = th.arange(batch_size, device=batch["mask"].device)

        no_info_batch["adjacency"][:, diagonal_indices, diagonal_indices] = 1
        no_info_batch["w_adjacency"][:, diagonal_indices, diagonal_indices] = 1

        top1_idx, top2_idx = self._construct_predictions(
            batch=no_info_batch,
            diagonal_indices=diagonal_indices,
            batch_indices=batch_indices,
        )
        return (
            no_info_batch["adjacency"],
            no_info_batch["w_adjacency"],
            target_t_adjacency,
            target_w_adjacency,
            top1_idx,
            top2_idx,
        )

    @th.no_grad()
    def _construct_predictions(
        self,
        batch: SequenceBatch,
        diagonal_indices: th.Tensor,
        batch_indices: th.Tensor,
    ) -> tuple[TopIndexCollection, TopIndexCollection]:
        seq_len = len(diagonal_indices)

        w_gain, pair_output = self._construct_w_gain(batch, diagonal_indices)
        idx_w_11, idx_w_12 = th.unravel_index(
            th.vmap(th.argmax)(w_gain),
            (seq_len, seq_len),
        )
        batch["w_adjacency"] = self._insert_edges(
            batch["w_adjacency"],
            idx_w_11,
            idx_w_12,
        )
        batch["adjacency"] = self._insert_edges(
            batch["adjacency"],
            idx_w_11,
            idx_w_12,
        )
        batch["mask_adjacency"] = self._insert_first_w(
            batch["mask_adjacency"],
            batch_indices,
            idx_w_11,
            idx_w_12,
        )
        # Update w_gain if we're doing discrete diffusion
        if self._use_discrete_diffusion:
            w_gain, _ = self._construct_w_gain(batch, diagonal_indices)

        # mask out predicted w edges
        w_gain[batch_indices, idx_w_11, :] = -th.inf
        w_gain[batch_indices, idx_w_12, :] = -th.inf
        w_gain[batch_indices, :, idx_w_11] = -th.inf
        w_gain[batch_indices, :, idx_w_12] = -th.inf

        # Predict second W-boson
        idx_w_21, idx_w_22 = th.unravel_index(
            th.vmap(th.argmax)(w_gain),
            (seq_len, seq_len),
        )
        batch["w_adjacency"] = self._insert_edges(
            batch["w_adjacency"],
            idx_w_21,
            idx_w_22,
        )
        batch["adjacency"] = self._insert_edges(
            batch["adjacency"],
            idx_w_21,
            idx_w_22,
        )
        batch["mask_adjacency"] = self._insert_second_w(
            batch["mask_adjacency"],
            batch_indices,
            idx_w_11,
            idx_w_12,
            idx_w_21,
            idx_w_22,
        )

        if self._use_discrete_diffusion:
            gain_from_w1, gain_from_w2 = self._compute_gain_from_ws(
                batch,
                idx_w_11=idx_w_11,
                idx_w_12=idx_w_12,
                idx_w_21=idx_w_21,
                idx_w_22=idx_w_22,
                batch_indices=batch_indices,
            )
        else:
            diagonal_contribution = th.vmap(th.diag)(pair_output[..., 0])
            gain_from_w1 = (
                pair_output[batch_indices, idx_w_11, :, 0]
                + pair_output[batch_indices, :, idx_w_12, 0]
                - diagonal_contribution
            )
            gain_from_w2 = (
                pair_output[batch_indices, idx_w_21, :, 0]
                + pair_output[batch_indices, :, idx_w_22, 0]
                - diagonal_contribution
            )

        # Mask previously selected edges
        for idx_loc1 in [idx_w_11, idx_w_12, idx_w_21, idx_w_22]:
            gain_from_w1[batch_indices, idx_loc1] = -th.inf
            gain_from_w2[batch_indices, idx_loc1] = -th.inf

        if self._use_split_b_prediction:
            gain1_values, gain1_idx = th.max(gain_from_w1, dim=-1)
            gain2_values, gain2_idx = th.max(gain_from_w2, dim=-1)

            chosen_w1_first = gain1_values > gain2_values
            chosen_w2_first = ~chosen_w1_first

            chosen_w1_batch_indices = th.arange(
                th.sum(chosen_w1_first), device=chosen_w1_first.device
            )
            chosen_w2_batch_indices = th.arange(
                th.sum(chosen_w2_first), device=chosen_w2_first.device
            )

            batch["adjacency"][chosen_w1_first] = self._insert_top_edges(
                batch["adjacency"][chosen_w1_first],
                b_idx=gain1_idx[chosen_w1_first],
                w_idx1=idx_w_11[chosen_w1_first],
                w_idx2=idx_w_12[chosen_w1_first],
                batch_indices=chosen_w1_batch_indices,
            )
            batch["adjacency"][~chosen_w1_first] = self._insert_top_edges(
                batch["adjacency"][chosen_w2_first],
                b_idx=gain2_idx[chosen_w2_first],
                w_idx1=idx_w_21[chosen_w2_first],
                w_idx2=idx_w_22[chosen_w2_first],
                batch_indices=chosen_w2_batch_indices,
            )
            batch["mask_adjacency"] = self._insert_top(
                batch["mask_adjacency"],
                b_idx=gain1_idx,
                w_idx1=idx_w_11,
                w_idx2=idx_w_12,
                top_mask=chosen_w1_first,
            )
            batch["mask_adjacency"] = self._insert_top(
                batch["mask_adjacency"],
                b_idx=gain2_idx,
                w_idx1=idx_w_21,
                w_idx2=idx_w_22,
                top_mask=chosen_w2_first,
            )
            # Second b-quark prediction
            second_gain_from_w1, second_gain_from_w2 = self._compute_gain_from_ws(
                batch,
                idx_w_11=idx_w_11,
                idx_w_12=idx_w_12,
                idx_w_21=idx_w_21,
                idx_w_22=idx_w_22,
                batch_indices=batch_indices,
            )
            # Mask previously selected edges
            for idx_loc1 in [idx_w_11, idx_w_12, idx_w_21, idx_w_22]:
                second_gain_from_w1[batch_indices, idx_loc1] = -th.inf
                second_gain_from_w2[batch_indices, idx_loc1] = -th.inf

            # Ensure that the same b-quark is not selected twice
            second_gain_from_w1[chosen_w2_first, gain2_idx[chosen_w2_first]] = -th.inf
            second_gain_from_w2[chosen_w1_first, gain1_idx[chosen_w1_first]] = -th.inf
            second_gain1_values, second_gain1_idx = th.max(second_gain_from_w1, dim=-1)
            second_gain2_values, second_gain2_idx = th.max(second_gain_from_w2, dim=-1)

            batch["adjacency"][chosen_w2_first] = self._insert_top_edges(
                batch["adjacency"][chosen_w2_first],
                b_idx=second_gain1_idx[chosen_w2_first],
                w_idx1=idx_w_11[chosen_w2_first],
                w_idx2=idx_w_12[chosen_w2_first],
                batch_indices=chosen_w2_batch_indices,
            )
            batch["adjacency"][chosen_w1_first] = self._insert_top_edges(
                batch["adjacency"][chosen_w1_first],
                b_idx=second_gain2_idx[chosen_w1_first],
                w_idx1=idx_w_21[chosen_w1_first],
                w_idx2=idx_w_22[chosen_w1_first],
                batch_indices=chosen_w1_batch_indices,
            )

            top1_idx = TopIndexCollection(
                idx_b=th.where(chosen_w1_first, gain1_idx, second_gain1_idx),
                idx_w1=idx_w_11,
                idx_w2=idx_w_12,
            )
            top2_idx = TopIndexCollection(
                idx_b=th.where(chosen_w2_first, gain2_idx, second_gain2_idx),
                idx_w1=idx_w_21,
                idx_w2=idx_w_22,
            )

        else:
            b_gain = gain_from_w1.unsqueeze(-1) + gain_from_w2.unsqueeze(-2)
            # Avoid assigning same b quark to both w.
            b_gain.diagonal(dim1=-2, dim2=-1).fill_(-th.inf)
            idx_b1, idx_b2 = th.unravel_index(
                th.vmap(th.argmax)(b_gain),
                (seq_len, seq_len),
            )
            batch["adjacency"] = self._insert_top_edges(
                batch["adjacency"],
                b_idx=idx_b1,
                w_idx1=idx_w_11,
                w_idx2=idx_w_12,
                batch_indices=batch_indices,
            )
            batch["adjacency"] = self._insert_top_edges(
                batch["adjacency"],
                b_idx=idx_b2,
                w_idx1=idx_w_21,
                w_idx2=idx_w_22,
                batch_indices=batch_indices,
            )
            top1_idx = TopIndexCollection(
                idx_b=idx_b1,
                idx_w1=idx_w_11,
                idx_w2=idx_w_12,
            )
            top2_idx = TopIndexCollection(
                idx_b=idx_b2,
                idx_w1=idx_w_21,
                idx_w2=idx_w_22,
            )
        return top1_idx, top2_idx

    def _insert_edges(
        self,
        adjacency_matrix: th.Tensor,
        idx_1: th.Tensor,
        idx_2: th.Tensor,
    ) -> th.Tensor:
        batch_size, seq_len, _ = adjacency_matrix.shape
        batch_indices = th.arange(batch_size, device=adjacency_matrix.device)
        adjacency_matrix[batch_indices, idx_1, idx_2] = 1
        adjacency_matrix[batch_indices, idx_2, idx_1] = 1

        adjacency_matrix[batch_indices, idx_1, idx_1] = 0
        adjacency_matrix[batch_indices, idx_2, idx_2] = 0
        return adjacency_matrix

    def _insert_top(
        self,
        adjacency_mask: th.Tensor,
        b_idx: th.Tensor,
        w_idx1: th.Tensor,
        w_idx2: th.Tensor,
        top_mask: th.Tensor,
    ):
        """Utility to update adjacency mask with predicted b-quark connections.

        Args:
            adjacency_mask: Tensor of shape (batch_size, seq_len, seq_len, 2)
            b_idx: Indices of predicted b-quarks.
            w_idx1: Indices of first W-boson quark.
            w_idx2: Indices of second W-boson quark.
            top_mask: Mask indicating which entries to update.

        Returns:
            Updated adjacency mask.
        """
        adjacency_mask[top_mask, b_idx[top_mask], :, 0] = True
        adjacency_mask[top_mask, :, b_idx[top_mask], 0] = True
        adjacency_mask[top_mask, b_idx[top_mask], w_idx1[top_mask], 0] = True
        adjacency_mask[top_mask, w_idx1[top_mask], b_idx[top_mask], 0] = True
        adjacency_mask[top_mask, b_idx[top_mask], w_idx2[top_mask], 0] = True
        adjacency_mask[top_mask, w_idx2[top_mask], b_idx[top_mask], 0] = True
        return adjacency_mask

    def _insert_first_w(
        self,
        adjacency_matrix: th.Tensor,
        batch_indices: th.Tensor,
        idx_1: th.Tensor,
        idx_2: th.Tensor,
    ) -> th.Tensor:
        adjacency_matrix[batch_indices, idx_1, :, 1] = True
        adjacency_matrix[batch_indices, :, idx_1, 1] = True
        adjacency_matrix[batch_indices, idx_2, :, 1] = True
        adjacency_matrix[batch_indices, :, idx_2, 1] = True
        adjacency_matrix[batch_indices, idx_1, idx_1, 0] = True
        adjacency_matrix[batch_indices, idx_1, idx_2, 0] = True
        adjacency_matrix[batch_indices, idx_2, idx_1, 0] = True
        adjacency_matrix[batch_indices, idx_2, idx_2, 0] = True
        return adjacency_matrix

    def _insert_second_w(
        self,
        adjacency_matrix: th.Tensor,
        batch_indices: th.Tensor,
        idx_w_11: th.Tensor,
        idx_w_12: th.Tensor,
        idx_w_21: th.Tensor,
        idx_w_22: th.Tensor,
    ) -> th.Tensor:
        adjacency_matrix[..., 1] = True
        for idx_loc1, idx_loc2 in product(
            [idx_w_11, idx_w_12, idx_w_21, idx_w_22], repeat=2
        ):
            adjacency_matrix[batch_indices, idx_loc1, idx_loc2, 0] = True
        return adjacency_matrix

    def _insert_top_edges(
        self,
        adjacency_matrix: th.Tensor,
        b_idx: th.Tensor,
        w_idx1: th.Tensor,
        w_idx2: th.Tensor,
        batch_indices: th.Tensor,
    ) -> th.Tensor:
        # Remove self-loop
        adjacency_matrix[batch_indices, b_idx, b_idx] = 0

        # Connect chosen b-quark to existing W-cluster
        adjacency_matrix[batch_indices, b_idx, w_idx1] = 1
        adjacency_matrix[batch_indices, w_idx1, b_idx] = 1
        adjacency_matrix[batch_indices, b_idx, w_idx2] = 1
        adjacency_matrix[batch_indices, w_idx2, b_idx] = 1
        return adjacency_matrix

    def _compute_gain_from_ws(
        self,
        batch: SequenceBatch,
        idx_w_11: th.Tensor,
        idx_w_12: th.Tensor,
        idx_w_21: th.Tensor,
        idx_w_22: th.Tensor,
        batch_indices: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        latent_sequences, latent_pairs, mask = self(batch)
        if self._output_hypergraph:
            logits_3d = self._output_3d_func(latent_sequences).squeeze(-1)
            gain_from_w1 = logits_3d[batch_indices, idx_w_11, idx_w_12]
            gain_from_w2 = logits_3d[batch_indices, idx_w_21, idx_w_22]
        else:
            top_logits = self._output_pair_func(latent_sequences, latent_pairs)[..., 0]
            diagonal_contribution = th.vmap(th.diag)(top_logits)
            gain_from_w1 = (
                top_logits[batch_indices, idx_w_11, :]
                + top_logits[batch_indices, :, idx_w_12]
                - diagonal_contribution
            )
            gain_from_w2 = (
                top_logits[batch_indices, idx_w_21, :]
                + top_logits[batch_indices, :, idx_w_22]
                - diagonal_contribution
            )
        gain_from_w1[~mask] = -th.inf
        gain_from_w2[~mask] = -th.inf
        return gain_from_w1, gain_from_w2

    def _construct_w_gain(
        self,
        batch: SequenceBatch,
        diagonal_indices: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        latent_sequences, latent_pairs, mask = self(batch)
        pair_output = self._output_pair_func(
            latent_sequences,
            latent_pairs,
        )
        w_output_matrix = pair_output[..., -1]
        w_gain = th.vmap(compute_gain_from_pair_logits)(
            self._compute_w_logits(w_output_matrix, mask)
        )
        w_gain[:, diagonal_indices, diagonal_indices] = -th.inf  # No self-loops
        mask = batch["mask"]
        mask_matrix = construct_mask_matrix(mask)
        w_gain[~mask_matrix] = -th.inf  # Mask out invalid entries
        return w_gain, pair_output

    def _compute_w_logits(
        self,
        w_output_matrix: th.Tensor,
        mask: th.Tensor,
    ) -> th.Tensor:
        if self._sinkhorn_knopp_iterations:
            mask_matrix = construct_mask_matrix(mask)
            pre_probababilities = th.exp(w_output_matrix * mask_matrix)
            w_probabilities = sinkhorn_knopp(
                pre_probababilities,
                dim1=1,
                dim2=2,
                num_iters=self._sinkhorn_knopp_iterations,
            )
            return th.logit(w_probabilities)
        return w_output_matrix
