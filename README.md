# Pairton

Pairton is a deep learning framework for jet combinatorics in high-energy physics, focused on reconstructing top quark decays in ttbar events. It uses a [PairFormer](https://www.nature.com/articles/s41586-024-07487-w)-based architecture to jointly predict pairwise (W boson, 2-jet) and hyperedge (top quark, 3-jet) assignments from reconstructed jet sequences.

## Overview

In ttbar events, each top quark decays into three jets (one b-jet and two light jets from the W boson). Given a set of reconstructed jets, pairton predicts:

- **Pair adjacency** – which two jets originate from a W boson decay.
- **Hyper-edge adjacency** – which three jets form a top quark candidate.

The model encodes jets as single-element features and constructs pairwise representations, which are processed by a PairFormer stack before separate prediction heads produce the final edge and hyper-edge logits.

### Key features

- **PairFormer backbone** with configurable depth and dimensions for both single and pair representations.
- **HyperEdge prediction head** for direct 3-jet group prediction without enumerating all triplets at inference time.
- **Multiple training modes**: standard cross-entropy, discrete diffusion, and split-diffusion (separate W and top predictions).
- **LightningCLI** interface with [jsonargparse / OmegaConf](https://jsonargparse.readthedocs.io/) for fully composable YAML configs.
- **Snakemake workflows** for reproducible multi-run experiments on HPC clusters (SLURM).
- **Weights & Biases** integration for experiment tracking and config logging.

## Repository structure

```
pairton/
├── configs/               # YAML configs for models, data, and trainer
│   ├── models/            # Model architecture configs
│   ├── data/              # Dataset and feature-transform configs
│   └── trainer/           # PyTorch Lightning trainer configs
├── scripts/
│   ├── hyper_to_h5_cli.py # Convert ROOT files → HDF5
│   └── uncertainty_computation.py
├── src/pairton/
│   ├── data/              # DataModule, dataset factories, transforms
│   ├── layers/            # HyperEdgeLayer, symmetric layers
│   ├── models/            # PairHyperModel and wrapper models
│   └── main.py            # LightningCLI entry point
├── tests/
├── workflow/              # Snakemake workflows for experiment pipelines
└── pyproject.toml
```

## Installation

Pairton uses [uv](https://github.com/astral-sh/uv) for dependency management and requires **Python 3.12**.

```bash
# Install uv if you haven't already
pip install uv

# Create a virtual environment and install all dependencies
uv sync

# Activate the environment
source .venv/bin/activate
```

The `root` optional dependency group adds support for reading ROOT files (`uproot`, `awkward`):

```bash
uv sync --extra root
```

## Data preparation

Pairton reads data from HDF5 files. A helper script converts Delphes ROOT files to the expected format:

```bash
python scripts/hyper_to_h5_cli.py \
    --config <variable_config.yaml> \
    --tree Delphes \
    --output ttbar_train.h5 \
    input.root
```

The variable config YAML defines which branches to extract and how to group them (jets, truth-match info, etc.).

## Training

Training is driven by the `pairton` CLI, which is a `LightningCLI` wrapper:

```bash
pairton fit \
    --model configs/models/pairformer_hyper.yaml \
    --data  configs/data/sequence_hyper.yaml \
    --trainer configs/trainer/default_trainer.yaml
```

Individual config values can be overridden on the command line:

```bash
pairton fit \
    --model configs/models/pairformer_hyper.yaml \
    --data  configs/data/sequence_hyper.yaml \
    --trainer configs/trainer/default_trainer.yaml \
    --trainer.max_epochs=50 \
    --trainer.precision=bf16-mixed
```

## Inference / Prediction

```bash
pairton predict \
    --config predict_config.yaml \
    --ckpt_path /path/to/checkpoint.ckpt
```

## Experiment workflows

Snakemake workflows under `workflow/` orchestrate multi-model comparison experiments. Each workflow reads a YAML config (e.g. `workflow/config/diffusion_comparison.yaml`) that specifies output paths, model/data/trainer config names, W&B project details, and SLURM resource requirements.

```bash
# Run from the repo root, pointing at the desired workflow config
snakemake -s workflow/diffusion_comparison.smk \
    --executor slurm \
    --jobs 8
```

Available workflows:

| Workflow | Description |
|---|---|
| `diffusion_comparison.smk` | Compares standard, diffusion, and split-diffusion training modes |
| `model_comparison.smk` | Compares different model architectures |
| `get_uncertainties.smk` | Runs uncertainty estimation over saved checkpoints |

## Development

```bash
# Run tests
pytest tests/

# Lint
ruff check src/ tests/

# Type-check
pyright src/
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

Andreas Hermansen – [mail@andreashermansen.dk](mailto:mail@andreashermansen.dk)
