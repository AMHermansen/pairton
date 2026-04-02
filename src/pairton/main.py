from typing import override

from lightning import LightningModule
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from pairton.data.data_module import SequenceDataModule
from ml_utils.lightning_utils import WandBSaveConfigCallback


class SequenceCLI(LightningCLI):
    def __init__(self):
        """Constructs a LightningCLI for the FlatModel with SequenceDataModule.

        This CLI is used to finetune data using the flat representation.
        """
        super().__init__(
            model_class=LightningModule,  # combinatorics.models.base_sequence.BaseSequenceModel
            datamodule_class=SequenceDataModule,
            save_config_callback=WandBSaveConfigCallback,
            save_config_kwargs={"overwrite": True},
            parser_kwargs={"parser_mode": "omegaconf"},
            subclass_mode_model=True,
        )

    # Hack to disable checkpoint hyperparameter parsing.
    @override
    def _parse_ckpt_path(self) -> None:
        pass

    @override
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments(
            "data.dataset_factory.num_features",
            "model.init_args.in_features",
            apply_on="instantiate",
        )


def main():
    SequenceCLI()


if __name__ == "__main__":
    main()
