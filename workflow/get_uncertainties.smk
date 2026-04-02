from pathlib import Path

from pairton.models.callbacks import add_hdf5_writer_callback_command
from ml_utils.snakemake import add_best_checkpoint_command, add_csv_logger_command, \
    add_early_stopping_command, \
    add_learning_rate_monitor_command, add_snakemake_confirm_callback_command, \
    add_wandb_logger_command

configfile: "/home/users/h/hermanse/workspace/pairton/workflow/config/get_uncertainties.yaml"
base_output_dir = Path(config["output_dir"])
package_dir = Path("/home/users/h/hermanse/workspace/pairton/")

config_dir = package_dir / "configs"

activate_venv_cmd = f"source {package_dir}/.venv/bin/activate"

model_name = f"{config['model_config_name']}.yaml"
data_name = f"{config['data_config_name']}.yaml"
data_inference_name = f"{config['data_inference_config_name']}.yaml"
trainer_name = f"{config['trainer_config_name']}.yaml"

model_config_path = config_dir / "models" / model_name
data_config_path_normal = config_dir / "data" / data_name
data_config_path_dif3 = config_dir / "data" / f"{config['data_config_name']}_dif3.yaml"
data_config_path_inference = config_dir / "data" / data_inference_name
trainer_config_path = config_dir / "trainer" / trainer_name

train_out_dir = (
        base_output_dir
        / model_name.split(".", maxsplit=1)[0]
        / data_name.split(".", maxsplit=1)[0]
        / trainer_name.split(".", maxsplit=1)[0]
        / "uncertainties"
)

project_name = config["project_name"]

metric_name = config["metric_name"]
mode = config["mode"]

train_command = config["train_command"]
test_command = config["test_command"]
predict_command = config["predict_command"]

seeds = config["seeds"]

normal_train_out_dirs = [train_out_dir / f"seed{seed}" for seed in seeds]

diffusion_toggle_cmd = "--model.init_args.diffusion_config.use_discrete_diffusion=true"

rule all:
    input:
        *[out_dir / "confirm.txt" for out_dir in normal_train_out_dirs],


for seed, out_dir in zip(seeds, normal_train_out_dirs):
    rule:
        name: f"performance_seed{seed}"
        output:
            out_dir / "confirm.txt",  # Dummy output to indicate completed training
        params:
            out_config_path = out_dir / "logs" / "config.yaml",
            out_ckpt_path = out_dir / "ckpts" / "best.ckpt",
            seed = seed,
            model_config_path = model_config_path,
            data_config_path = data_config_path_normal,
            trainer_config_path = trainer_config_path,
            wandb_logger_command = add_wandb_logger_command(
                project=project_name,
                name=f"{model_name.split('.')[0]}_seed{seed}",
                save_dir=out_dir / "logs",
                tags=[model_name, data_name, trainer_name, "train", f"diffusion_comparison"],
            ),
            csv_logger_command = add_csv_logger_command(
                save_dir=out_dir / "logs"
            ),
            best_checkpoint_command = add_best_checkpoint_command(
                dir_path=out_dir / "ckpts",
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
                out_dir / "confirm.txt"
            ),
            hdf5_saver_command = add_hdf5_writer_callback_command(
                out_dir / "predictions" / "predictions.hdf5"
            )
        shell:
            f"""
                {activate_venv_cmd}
                
                {train_command} \
                    --model={{params.model_config_path}} \
                    {diffusion_toggle_cmd} \
                    --data={{params.data_config_path}} \
                    --trainer={{params.trainer_config_path}} \
                    {{params.wandb_logger_command}} \
                    {{params.best_checkpoint_command}} \
                    {{params.learning_rate_command}} \
                    {{params.snakemake_confirm_command}} \
                    --seed={{params.seed}}
                    
                {test_command} --config {{params.out_config_path}} --data={data_config_path_inference} --ckpt_path {{params.out_ckpt_path}} {{params.csv_logger_command}}
                {predict_command} --config {{params.out_config_path}} --data={data_config_path_inference} --ckpt_path {{params.out_ckpt_path}} {{params.hdf5_saver_command}} 
                """
