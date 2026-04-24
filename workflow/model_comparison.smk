from pathlib import Path

from ml_utils.snakemake import (
    add_best_checkpoint_command,
    add_csv_logger_command,
    add_early_stopping_command,
    add_learning_rate_monitor_command,
    add_snakemake_confirm_callback_command,
    add_wandb_logger_command,

)

configfile: "/home/users/h/hermanse/workspace/pairton/workflow/config/model_comparison.yaml"
base_output_dir = Path(config["output_dir"])
package_dir = Path("/home/users/h/hermanse/workspace/pairton/")

config_dir = package_dir / "configs"

activate_venv_cmd = f"source {package_dir}/.venv/bin/activate"

model_names = ["bias_transformer.yaml", "transformer.yaml"]
data_name = f"{config['data_config_name']}.yaml"
data_inference_name = f"{config['data_inference_config_name']}.yaml"
trainer_name = f"{config['trainer_config_name']}.yaml"

model_config_paths = [config_dir / "models" / name for name in model_names]
data_config_path = config_dir / "data" / data_name
data_config_path_inference = config_dir / "data" / data_inference_name
trainer_config_path = config_dir / "trainer" / trainer_name

train_out_dirs = [
    base_output_dir
    / model_name.split(".")[0]
    / data_name.split(".", maxsplit=1)[0]
    / trainer_name.split(".", maxsplit=1)[0]
    for model_name in model_names
]

project_name = config["project_name"]

metric_name = config["metric_name"]
mode = config["mode"]

train_command = config["train_command"]
validate_command = config["validate_command"]

seed = config["seed"]

rule all:
    input :
        expand(
            str("{train_out_dir}/ckpts/best.ckpt"),
            train_out_dir=train_out_dirs
        )

for model_config_path, train_out_dir, model_name in zip(
        model_config_paths,
        train_out_dirs,
        model_names
):
    rule_name = f"train_{model_name.split('.')[0]}"
    rule:
        name: rule_name
        output:
            train_out_dir / "ckpts" / "best.ckpt"
        params:
            out_config_path = train_out_dir / "logs" / "config.yaml",
            out_ckpt_path = train_out_dir / "ckpts" / "best.ckpt",
            model_config_path = model_config_path,
            data_config_path = data_config_path,
            trainer_config_path = trainer_config_path,
            logger_command = add_wandb_logger_command(
                project=project_name,
                name=model_name,
                save_dir=train_out_dir / "logs",
                tags=[model_name, data_name, trainer_name, "train"],
            ),
            csv_logger_command = add_csv_logger_command(
                save_dir=train_out_dir / "logs",
            ),
            best_checkpoint_command = add_best_checkpoint_command(
            dir_path=train_out_dir / "ckpts",
                monitor=metric_name,
                mode=mode,
                ),
            early_stopping_command = add_early_stopping_command(
            monitor=metric_name,
                mode=mode,
                patience=5,
                ),
            learning_rate_command = add_learning_rate_monitor_command(),
            snakemake_confirm_command = add_snakemake_confirm_callback_command(
                train_out_dir / "confirm.txt"
            ),
        shell:
            f"""
            {activate_venv_cmd}
            {train_command} \
                --model={{params.model_config_path}} \
                --data={{params.data_config_path}} \
                --trainer={{params.trainer_config_path}} \
                {{params.logger_command}} \
                {{params.best_checkpoint_command}} \
                {{params.learning_rate_command}} \
                {{params.snakemake_confirm_command}} \
                --seed={seed} \
                
            {validate_command} --config {{params.out_config_path}} --data={data_config_path_inference} --ckpt_path={{params.out_ckpt_path}} {{params.csv_logger_command}}
            """
