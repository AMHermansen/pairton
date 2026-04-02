from collections.abc import Sequence
from pathlib import Path
from typing import Any, override

import h5py
import lightning.pytorch as pl
import torch as th
from lightning.pytorch.callbacks import BasePredictionWriter


class HDF5Writer(BasePredictionWriter):
    @override
    def __init__(
        self,
        output_file: Path | str,
    ):
        super().__init__(write_interval="epoch")
        self.output_file = Path(output_file)

    @override
    def setup(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: str | None = None,
    ) -> None:
        if not stage == "predict":
            return  # only set up for predict stage
        if self.output_file.exists():
            raise FileExistsError(f"The file {self.output_file} already exists.")
        if not self.output_file.parent.exists():
            self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Sequence[Any],
        batch_indices: Sequence[Any],
    ) -> None:
        all_group0 = [p["groups"][0] for p in predictions]
        all_group1 = [p["groups"][1] for p in predictions]
        all_top1_idx = [p["top1_idx"] for p in predictions]
        all_top2_idx = [p["top2_idx"] for p in predictions]

        all_top1_idx_combined = (
            th.cat([
                th.stack(
                    [
                        top_idx_collection.idx_b,
                        top_idx_collection.idx_w1,
                        top_idx_collection.idx_w2,
                    ],
                    dim=-1,
                )
                for top_idx_collection in all_top1_idx
            ])
            .cpu()
            .numpy()
        )
        all_top2_idx_combined = (
            th.cat(
                [
                    th.stack(
                        [
                            top_idx_collection.idx_b,
                            top_idx_collection.idx_w1,
                            top_idx_collection.idx_w2,
                        ],
                        dim=-1,
                    )
                    for top_idx_collection in all_top2_idx
                ],
                dim=0,
            )
            .cpu()
            .numpy()
        )

        all_group0_combined = (
            th.cat(
                all_group0,
                dim=0,
            )
            .cpu()
            .numpy()
        )
        all_group1_combined = (
            th.cat(
                all_group1,
                dim=0,
            )
            .cpu()
            .numpy()
        )

        with h5py.File(self.output_file, "w") as h5_file:
            h5_file.create_dataset("truth_group0", data=all_group0_combined)
            h5_file.create_dataset("truth_group1", data=all_group1_combined)
            h5_file.create_dataset("predict_top0_idx", data=all_top1_idx_combined)
            h5_file.create_dataset("predict_top1_idx", data=all_top2_idx_combined)


def add_hdf5_writer_callback_command(
    output_file: Path | str,
) -> str:
    return " ".join([
        f"--trainer.callbacks+={HDF5Writer.__module__}.{HDF5Writer.__name__}",
        f"--trainer.callbacks.output_file={output_file}",
    ])
