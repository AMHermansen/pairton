PairTON
======

Source code for PairTON.  <!-- TODO: Insert arxiv link -->

Snakemake workflows
-------------------

Execute the repository's Snakemake workflows with the following command (example):

```bash
snakemake -s workflow/workflow_name.smk --workflow-profile workflow/profile/
```

There are three separate workflows, one for running the main configuration and getting uncertainties,
and then two workflows the architecture ablation and iteration ablation studies.

Data & configuration
---------------------

Configuration files live under `configs/` (for example `configs/data/sequence_hyper.yaml`).
Some configuration files contain absolute, hardcoded dataset paths that must be adapted to your system before running the workflows.

Warning: hardcoded paths
------------------------

This repository currently contains absolute paths that start with `/srv/`. Before running anything, search the tree and update paths as needed. For a quick check run:

```bash
grep -R --line-number "/srv/" .
```

Replace occurrences with paths valid on your system or use relative paths/configuration overrides.

License
-------

This project is distributed under the MIT License. See `LICENSE` for details.

Maintainer
----------

andreas.hermansen@unige.ch
