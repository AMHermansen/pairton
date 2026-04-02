from itertools import permutations, product
from typing import override

import einx
import torch as th
from lightning import LightningModule
from torch import nn

from pairton.data.types import SequenceBatch
from pairton.layers.hyper import HyperEdgeLayer
from pairton.layers.symmetric import MultiDimensionalLinearNoBiasThenOuterSum
from pairton.models.metrics import (
    DiffusionFullTPR,
    DiffusionTopTPR,
    DiffusionWTPR,
)
from pairton.models.utils import compute_pair_features
from ml_utils.components import MLP, SwiGLUMLP
from ml_utils.components.pairformer import PairFormer, PairFormerConfig
from ml_utils.lightning_utils import LightningConfig, configure_optimizer_standard
from ml_utils.utils import default


class PairHyperModel(LightningModule):
    INPUT_PAIR_FEATURES = 6
    ALPHA = 0.8

    def __init__(
        self,
        in_features: int,
        latent_dim_single: int = 128,
        latent_dim_pair: int = 64,
        pairformer_config: PairFormerConfig | None = None,
        *,
        lightning_config: LightningConfig | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self._in_features = in_features
        self._latent_dim_single = latent_dim_single
        self._latent_dim_pair = latent_dim_pair
        self._pairformer_config = pairformer_config
        self._lightning_config = default(lightning_config, LightningConfig())

        self._setup_model()

    @override
    def training_step(
        self,
        batch: SequenceBatch,
        batch_idx: int,
    ) -> dict[str, th.Tensor]:
        del batch_idx  # unused
        latent_sequence, pair_logits, hyper_logits, mask = self._share_step(batch)
        pair_loss, hyper_loss = self._compute_loss(
            pair_logits=pair_logits,
            hyper_logits=hyper_logits,
            batch=batch,
        )
        loss = self.ALPHA * hyper_loss + (1 - self.ALPHA) * pair_loss
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/pair_loss",
            pair_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/hyper_loss",
            hyper_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return {
            "loss": loss,
            "pair_logits": pair_logits.detach(),
            "hyper_logits": hyper_logits.detach(),
        }

    @override
    def validation_step(
        self,
        batch: SequenceBatch,
        batch_idx: int,
    ) -> dict[str, th.Tensor]:
        del batch_idx
        latent_sequence, pair_logits, hyper_logits, mask = self._share_step(batch)
        pair_loss, hyper_loss = self._compute_loss(
            pair_logits=pair_logits,
            hyper_logits=hyper_logits,
            batch=batch,
        )
        loss = self.ALPHA * hyper_loss + (1 - self.ALPHA) * pair_loss
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/pair_loss", pair_loss, prog_bar=True)
        self.log("val/hyper_loss", hyper_loss, prog_bar=True)

        adjacency_predict = self._predict_adjacency(
            pair_logits=pair_logits,
            hyper_logits=hyper_logits,
            mask=mask,
        )
        self._event_tpr_metric(
            t_adjacency_guess=adjacency_predict[..., 0],
            t_adjacency_target=batch["adjacency"],
            w_adjacency_guess=adjacency_predict[..., 1],
            w_adjacency_target=batch["w_adjacency"],
            mask=mask,
            groups=batch["groups"],
        )
        self._val_w_tpr_metric(
            adjacency_predict[..., 1],
            batch["w_adjacency1"],
            batch["w_adjacency2"],
            mask,
            batch["groups"],
        )
        self._val_t_tpr_metric(
            adjacency_predict[..., 0],
            adjacency_predict[..., 1],
            batch["top_adjacency1"],
            batch["w_adjacency1"],
            batch["top_adjacency2"],
            batch["w_adjacency2"],
            mask,
            batch["groups"],
        )
        self.log(
            "val/event_full_diffusion_tpr",
            self._event_tpr_metric,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/t_diffusion_tpr",
            self._val_t_tpr_metric,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/w_diffusion_tpr",
            self._val_w_tpr_metric,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return {
            "loss": loss,
            "pair_logits": pair_logits.detach(),
            "hyper_logits": hyper_logits.detach(),
            "adjacency_predict": adjacency_predict.detach(),
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

    # Model setup
    def _setup_model(self):
        self._pairformer = PairFormer(
            single_features=self._latent_dim_single,
            pair_features=self._latent_dim_pair,
            config=self._pairformer_config,
        )
        self._single_embed = nn.Sequential(
            nn.Linear(
                in_features=self._in_features,
                out_features=self._latent_dim_single,
                bias=False,
            ),
            nn.SiLU(),
            SwiGLUMLP(in_features=self._latent_dim_single),
        )
        self._post_model_sequence_norm = nn.RMSNorm(self._latent_dim_single)
        self._post_model_pair_norm = nn.RMSNorm(self._latent_dim_pair)
        self._sequence_to_pair_embed = nn.Sequential(
            MultiDimensionalLinearNoBiasThenOuterSum(
                dim=self._in_features,
                repeats=[1, 1],
                dim_out=self._latent_dim_pair,
            ),
            nn.LayerNorm(self._latent_dim_pair),
            nn.SiLU(),
            SwiGLUMLP(in_features=self._latent_dim_pair),
        )
        self._pair_to_pair_embed = MLP(
            in_features=self.INPUT_PAIR_FEATURES,
            out_features=self._latent_dim_pair,
        )
        self._single_to_hyper_proj = MultiDimensionalLinearNoBiasThenOuterSum(
            dim=self._latent_dim_single,
            repeats=[3],
        )
        self._hyper_predict = HyperEdgeLayer(
            in_features=self._latent_dim_single,
        )
        self._edge_predict = nn.Sequential(
            MultiDimensionalLinearNoBiasThenOuterSum(
                dim=self._latent_dim_single,
                repeats=[2],
            ),
            nn.SiLU(),
            nn.Linear(self._latent_dim_single, self._latent_dim_single, bias=False),
            nn.SiLU(),
            nn.Linear(self._latent_dim_single, 1, bias=False),
        )
        self._event_tpr_metric = DiffusionFullTPR()
        self._val_w_tpr_metric = DiffusionWTPR()
        self._val_t_tpr_metric = DiffusionTopTPR()

    # Utility methods for training/inference
    def _share_step(
        self,
        batch: SequenceBatch,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        single_representations = self._single_embed(batch["features"])
        mask = batch["mask"]

        pairwise_from_single_representations = self._sequence_to_pair_embed(
            batch["features"],
        )
        pairwise_from_pair_representations = self._pair_to_pair_embed(
            self._construct_pair_features(batch["features"])
        )
        total_pairwise_representations = (
            pairwise_from_single_representations + pairwise_from_pair_representations
        )
        (
            single_latent,
            pair_latent,
        ) = self._pairformer(
            single_features=single_representations,
            pair_features=total_pairwise_representations,
            mask=mask,
        )
        single_latent = self._post_model_sequence_norm(single_latent)
        # pair_latent = self._post_model_pair_norm(pair_latent)

        pair_logits = self._edge_predict(single_latent).squeeze(-1)
        hyper_logits = self._hyper_predict(
            self._single_to_hyper_proj(single_latent), mask
        )
        return single_latent, pair_logits, hyper_logits, mask

    def _construct_pair_features(self, features: th.Tensor) -> th.Tensor:
        """Args:
            features: Single features of shape (batch, seq_len, 6). These are
            expected to be
            - log(E) - 4.5
            - log(PT) - 4.5
            - eta
            - cos(phi)
            - sin(phi)
            - is_tagged.
            - ...
            If this is not the case, the returned values may not be meaningful!

        Returns:
            Pair features of shape (batch, seq_len, seq_len, 6).

        """
        return compute_pair_features(features)

    def _compute_loss(
        self,
        pair_logits: th.Tensor,
        hyper_logits: th.Tensor,
        batch: SequenceBatch,
    ) -> tuple[th.Tensor, th.Tensor]:
        w_adjacency = batch["w_adjacency"]
        max_num_nodes = w_adjacency.shape[1]  # b, num_nodes, num_nodes

        hyper_target = self._construct_hyper_target(
            groups=batch["groups"], max_num_nodes=max_num_nodes
        )

        pair_loss = nn.functional.binary_cross_entropy_with_logits(
            pair_logits,
            w_adjacency.to(pair_logits.dtype),
            reduction="none",
        )
        hyper_loss = nn.functional.binary_cross_entropy_with_logits(
            hyper_logits,
            hyper_target.to(hyper_logits.dtype),
            reduction="none",
        )
        edge_mask = einx.logical_and("b i, b j -> b i j", batch["mask"], batch["mask"])
        hyper_mask = self._construct_hyper_edge_mask(batch["mask"])
        pair_loss = (
            (
                (pair_loss * edge_mask.to(pair_loss.dtype))
                / edge_mask.sum(
                    dim=(1, 2),
                    keepdims=True,
                )
            )
            .sum(-1)
            .sum(-1)
        )
        hyper_loss = (
            (
                (hyper_loss * hyper_mask.to(hyper_loss.dtype))
                / hyper_mask.sum(
                    dim=(1, 2, 3),
                    keepdims=True,
                )
            )
            .sum(-1)
            .sum(-1)
            .sum(-1)
        )
        return pair_loss.mean(), hyper_loss.mean()

    @staticmethod
    def _construct_hyper_target(
        groups: list[th.Tensor],
        max_num_nodes: int,
    ):
        batch_size = groups[0].shape[0]
        batch_index = th.arange(batch_size, device=groups[0].device)
        hyper_target = th.zeros(
            (batch_size, max_num_nodes, max_num_nodes, max_num_nodes),
            dtype=th.float32,
            device=groups[0].device,
        )
        for group in groups:
            for i, j, k in permutations(group.unbind(dim=-1)):
                hyper_target[batch_index, i, j, k] = 1.0

        return hyper_target

    def _predict_adjacency(
        self,
        pair_logits: th.Tensor,
        hyper_logits: th.Tensor,
        mask: th.Tensor,
    ) -> th.Tensor:
        hyper_mask = self._construct_hyper_edge_mask(mask)
        edge_mask = einx.logical_and("b i, b j -> b i j", mask, mask)
        hyper_logits.masked_fill_(~hyper_mask, float("-inf"))
        pair_logits.masked_fill_(~edge_mask, float("-inf"))

        # Set diagonal to -inf to prevent self-loops in W-prediction
        node_idx = th.arange(pair_logits.shape[1], device=pair_logits.device)
        pair_logits[:, node_idx, node_idx] = float("-inf")

        hyper_predict1 = th.unravel_index(
            th.vmap(th.argmax)(hyper_logits),
            hyper_logits.shape[1:],
        )
        batch_idx = th.arange(pair_logits.shape[0], device=pair_logits.device)
        for idx in hyper_predict1:
            hyper_logits[batch_idx, idx, :, :] = float("-inf")
            hyper_logits[batch_idx, :, idx, :] = float("-inf")
            hyper_logits[batch_idx, :, :, idx] = float("-inf")

        hyper_predict2 = th.unravel_index(
            th.vmap(th.argmax)(hyper_logits),
            hyper_logits.shape[1:],
        )

        edge_predict1 = self._select_best_matching_edge(
            pair_logits=pair_logits,
            hyper_predict=hyper_predict1,
        )
        edge_predict2 = self._select_best_matching_edge(
            pair_logits=pair_logits,
            hyper_predict=hyper_predict2,
        )

        return self._construct_adjacency_from_indices(
            hyper_idx_groups=(hyper_predict1, hyper_predict2),
            edge_idx_groups=(edge_predict1, edge_predict2),
            num_nodes=pair_logits.shape[1],
        )

    def _select_best_matching_edge(
        self,
        pair_logits: th.Tensor,
        hyper_predict: tuple[th.Tensor, th.Tensor, th.Tensor],
    ) -> tuple[th.Tensor, th.Tensor]:
        batch_idx = th.arange(pair_logits.shape[0], device=pair_logits.device)
        i, j, k = hyper_predict
        edge_logits_ij = pair_logits[batch_idx, i, j]
        edge_logits_ik = pair_logits[batch_idx, i, k]
        edge_logits_jk = pair_logits[batch_idx, j, k]
        edge_logits = th.stack(
            (edge_logits_ij, edge_logits_ik, edge_logits_jk),
            dim=-1,
        )
        best_edge_idx = th.argmax(edge_logits, dim=-1)
        idx1 = th.where(best_edge_idx == 0, i, th.where(best_edge_idx == 1, i, j))
        idx2 = th.where(best_edge_idx == 0, j, th.where(best_edge_idx == 1, k, k))
        return idx1, idx2

    def _construct_adjacency_from_indices(
        self,
        hyper_idx_groups: tuple[
            tuple[th.Tensor, th.Tensor, th.Tensor],
            tuple[th.Tensor, th.Tensor, th.Tensor],
        ],
        edge_idx_groups: tuple[
            tuple[th.Tensor, th.Tensor], tuple[th.Tensor, th.Tensor]
        ],
        num_nodes: int,
    ):
        batch_size = hyper_idx_groups[0][0].shape[0]
        batch_idx = th.arange(batch_size, device=hyper_idx_groups[0][0].device)
        adjacency = (
            th.eye(num_nodes, device=hyper_idx_groups[0][0].device, dtype=th.int32)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(batch_size, num_nodes, num_nodes, 2)
        ).clone()
        for hyper_idx_group in hyper_idx_groups:
            for idx1, idx2 in product(hyper_idx_group, repeat=2):
                adjacency[batch_idx, idx1, idx2, 0] ^= 1

        for edge_idx_group in edge_idx_groups:
            for idx1, idx2 in product(edge_idx_group, repeat=2):
                adjacency[batch_idx, idx1, idx2, 1] ^= 1

        return adjacency

    @staticmethod
    def _construct_hyper_edge_mask(
        mask: th.Tensor,
    ) -> th.Tensor:
        """Construct a mask for valid hyperedges based on the input mask for nodes.

        Args:
            mask: A boolean tensor of shape (batch_size, num_nodes) indicating valid nodes.

        Returns:
            A boolean tensor of shape (batch_size, num_nodes, num_nodes, num_nodes) indicating valid hyperedges.
        """
        base_hyper_mask = einx.logical_and(
            "b i, b j, b k -> b i j k",
            mask,
            mask,
            mask,
        )
        node_idx = th.arange(mask.shape[1], device=mask.device)
        i = node_idx.view(1, -1, 1, 1)
        j = node_idx.view(1, 1, -1, 1)
        k = node_idx.view(1, 1, 1, -1)
        directed_valid = th.logical_and((i < j), (j < k))
        return th.logical_and(base_hyper_mask, directed_valid)
