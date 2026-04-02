from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing the input CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default=None,
        help="Path to the directory where the output summary files will be saved.",
    )

    drop_keys = [
        "epoch",
        "step",
        "test/event_full_diffusion_tpr",
        "test/t_diffusion_tpr",
        "test/w_diffusion_tpr",
    ]
    metric_group_names = [
        "event_full_diffusion_tpr",
        "w_diffusion_tpr",
        "t_diffusion_tpr",
    ]

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if args.output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    all_files = input_dir.glob("*/logs/csv_logs/version_0/metrics.csv")

    uncertainty_data = [
        pd.read_csv(file).drop(labels=drop_keys, axis="columns") for file in all_files
    ]
    df_combined = pd.concat(uncertainty_data, ignore_index=True)

    metric_group_keys = {
        metric_group_name: [
            key for key in df_combined.columns if metric_group_name in key
        ]
        for metric_group_name in metric_group_names
    }
    group_performance_means = {
        metric_group_name: df_combined[metric_group_keys[metric_group_name]].mean(
            axis=0
        )
        for metric_group_name in metric_group_names
    }
    group_performance_stds = {
        metric_group_name: df_combined[metric_group_keys[metric_group_name]].std(axis=0)
        for metric_group_name in metric_group_names
    }

    for metric_group_name in metric_group_names:
        group_means = group_performance_means[metric_group_name]
        group_stds = group_performance_stds[metric_group_name]
        
        open(output_dir / f"{metric_group_name}_performance_summary.txt", "w").write(
            f"Performance Summary for {metric_group_name}:\n\n"
            + "\n".join(
                [
                    f"{metric_key}: Mean = {group_means[metric_key]:.6f}, Std = {group_stds[metric_key]:.6f}"
                    for metric_key in metric_group_keys[metric_group_name]
                ]
            )
        )


if __name__ == "__main__":
    main()
