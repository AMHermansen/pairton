from pathlib import Path

from ml_utils.snakemake import add_best_checkpoint_command, add_early_stopping_command, \
    add_learning_rate_monitor_command, add_snakemake_confirm_callback_command, \
    add_wandb_logger_command

configfile: "/home/users/h/hermanse/workspace/pairton/workflow/config/diffusion_comparison.yaml"
base_output_dir = Path(config["output_dir"])
package_dir = Path("/home/users/h/hermanse/workspace/pairton/")

config_dir = package_dir / "configs"

activate_venv_cmd = f"source {package_dir}/.venv/bin/activate"

model_name = f"{config['model_config_name']}.yaml"
data_name = f"{config['data_config_name']}.yaml"
trainer_name = f"{config['trainer_config_name']}.yaml"

model_config_path = config_dir / "models" / model_name
data_config_path_normal = config_dir / "data" / data_name
data_config_path_dif3 = config_dir / "data" / f"{config['data_config_name']}_dif3.yaml"
trainer_config_path = config_dir / "trainer" / trainer_name

train_out_dir = (
    base_output_dir
    / model_name.split(".", maxsplit=1)[0]
    / data_name.split(".", maxsplit=1)[0]
    / trainer_name.split(".", maxsplit=1)[0]
)

project_name = config["project_name"]

metric_name = config["metric_name"]
mode = config["mode"]

train_command = config["train_command"]
validate_command = config["validate_command"]

seed = config["seed"]

normal_train_out_dir = train_out_dir / "normal"
diffusion_train_out_dir = train_out_dir / "diffusion"
diffusion_next_step_prediction_train_out_dir = train_out_dir / "diffusion_next_step_prediction"
diffusion_split_b_prediction_train_out_dir = train_out_dir / "diffusion_split_b_prediction"
pair_diffusion_train_out_dir = train_out_dir / "pair_diffusion"

# 3x deep model, to compare diffusion with larger model, but similar compute.
normal_train_deep_out_dir = train_out_dir / "deep"
normal_train_deep_diffusion_out_dir = train_out_dir / "deep_diffusion"

# pairformer with HyPER prediction head, to compare prediction methods.
normal_train_hyper_out_dir = train_out_dir / "hyper"


rule all:
    input:
        normal_train_out_dir / "confirm.txt",
        diffusion_train_out_dir / "confirm.txt",
        diffusion_next_step_prediction_train_out_dir / "confirm.txt",
        diffusion_split_b_prediction_train_out_dir / "confirm.txt",
        normal_train_deep_out_dir / "confirm.txt",
        pair_diffusion_train_out_dir / "confirm.txt",
        normal_train_hyper_out_dir / "confirm.txt",
        normal_train_deep_diffusion_out_dir / "confirm.txt",

for out_dir, train_name in zip(
    [normal_train_out_dir, diffusion_train_out_dir, diffusion_next_step_prediction_train_out_dir, diffusion_split_b_prediction_train_out_dir, pair_diffusion_train_out_dir],
    ["normal", "diffusion", "diffusion_next_step_prediction", "diffusion_split_b_prediction", "pair_diffusion"],
):
    if train_name == "diffusion":
        diffusion_toggle_cmd = "--model.init_args.diffusion_config.use_discrete_diffusion=true"
    elif train_name == "pair_diffusion":
        diffusion_toggle_cmd = "--model.init_args.diffusion_config.use_discrete_diffusion=true --model.init_args.use_pair_output=true"
    elif train_name == "diffusion_next_step_prediction":
        diffusion_toggle_cmd = "--model.init_args.diffusion_config.use_discrete_diffusion=true --model.init_args.diffusion_config.predict_next_step_only=true"
    elif train_name == "diffusion_split_b_prediction":
        diffusion_toggle_cmd = f"--model.init_args.diffusion_config.use_discrete_diffusion=true --model.init_args.diffusion_config.predict_next_step_only=true --model.init_args.diffusion_config.use_split_b_prediction=true --data={data_config_path_dif3!s}"
    else:
        diffusion_toggle_cmd = "--model.init_args.diffusion_config.use_discrete_diffusion=false"

    rule:
        name: f"train_{train_name}"
        output:
            out_dir / "confirm.txt"  # Dummy output to indicate completed training
        params:
            model_config_path = model_config_path,
            diffusion_toggle_cmd = diffusion_toggle_cmd,
            data_config_path = data_config_path_normal,
            trainer_config_path = trainer_config_path,
            logger_command = add_wandb_logger_command(
                project=project_name,
                name=f"{train_name}_{model_name}",
                save_dir=out_dir / "logs",
                tags=[model_name, data_name, trainer_name, "train", train_name, f"diffusion_comparison"],
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
        shell:
            f"""
                {activate_venv_cmd}
                {train_command} \
                    --model={{params.model_config_path}} \
                    --data={{params.data_config_path}} \
                    {{params.diffusion_toggle_cmd}} \
                    --trainer={{params.trainer_config_path}} \
                    {{params.logger_command}} \
                    {{params.best_checkpoint_command}} \
                    {{params.learning_rate_command}} \
                    {{params.snakemake_confirm_command}} \
                    --seed={seed}
                """

diffusion_toggle_cmd = "--model.init_args.diffusion_config.use_discrete_diffusion=false"
deep_config_path = config_dir / "models" / f"{config['model_config_name']}_deep.yaml"
rule:
    name: "train_normal_deep"
    output:
        normal_train_deep_out_dir / "confirm.txt"  # Dummy output to indicate completed training
    params:
        model_config_path = deep_config_path,
        diffusion_toggle_cmd = diffusion_toggle_cmd,
        data_config_path = data_config_path_normal,
        trainer_config_path = trainer_config_path,
        logger_command = add_wandb_logger_command(
            project=project_name,
            name=f"normal_deep_{model_name}",
            save_dir=normal_train_deep_out_dir / "logs",
            tags=[model_name, data_name, trainer_name, "train", "normal_deep", f"diffusion_comparison"],
        ),
        best_checkpoint_command = add_best_checkpoint_command(
            dir_path=normal_train_deep_out_dir / "ckpts",
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
            normal_train_deep_out_dir / "confirm.txt"
        ),
    resources:
        constraint="COMPUTE_MODEL_RTX_3090_25G",  # More VRAM needed to for deeper model.
        runtime=2880,  # Same train compute.
    shell:
        # need bf16 for stability
        f"""
            {activate_venv_cmd}
            {train_command} \
                --model={{params.model_config_path}} \
                {{params.diffusion_toggle_cmd}} \
                --data={{params.data_config_path}} \
                --trainer={{params.trainer_config_path}} \
                {{params.logger_command}} \
                {{params.best_checkpoint_command}} \
                {{params.learning_rate_command}} \
                {{params.snakemake_confirm_command}} \
                --seed={seed} \
                --trainer.precision=bf16-mixed
            """


diffusion_toggle_cmd = "--model.init_args.diffusion_config.use_discrete_diffusion=true"
deep_config_path = config_dir / "models" / f"{config['model_config_name']}_deep.yaml"
rule:
    name: "train_diffusion_deep"
    output:
        normal_train_deep_diffusion_out_dir / "confirm.txt"  # Dummy output to indicate completed training
    params:
        model_config_path = deep_config_path,
        diffusion_toggle_cmd = diffusion_toggle_cmd,
        data_config_path = data_config_path_normal,
        trainer_config_path = trainer_config_path,
        logger_command = add_wandb_logger_command(
            project=project_name,
            name=f"normal_deep_{model_name}",
            save_dir=normal_train_deep_diffusion_out_dir / "logs",
            tags=[model_name, data_name, trainer_name, "train", "normal_deep", f"diffusion_comparison"],
        ),
        best_checkpoint_command = add_best_checkpoint_command(
            dir_path=normal_train_deep_diffusion_out_dir / "ckpts",
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
            normal_train_deep_diffusion_out_dir / "confirm.txt"
        ),
    resources:
        constraint="COMPUTE_MODEL_RTX_3090_25G",  # More VRAM needed to for deeper model.
        runtime=2880,  # Same train compute.
    shell:
        # need bf16 for stability
        f"""
            {activate_venv_cmd}
            {train_command} \
                --model={{params.model_config_path}} \
                {{params.diffusion_toggle_cmd}} \
                --data={{params.data_config_path}} \
                --trainer={{params.trainer_config_path}} \
                {{params.logger_command}} \
                {{params.best_checkpoint_command}} \
                {{params.learning_rate_command}} \
                {{params.snakemake_confirm_command}} \
                --seed={seed} \
                --trainer.precision=bf16-mixed
            """



hyper_config_path = config_dir / "models" / f"{config['model_config_name']}_hyper.yaml"
rule:
    name: "train_normal_hyper"
    output:
        normal_train_hyper_out_dir / "confirm.txt"  # Dummy output to indicate completed training
    params:
        model_config_path = hyper_config_path,
        data_config_path = data_config_path_normal,
        trainer_config_path = trainer_config_path,
        logger_command = add_wandb_logger_command(
            project=project_name,
            name=f"normal_hyper_{model_name}",
            save_dir=normal_train_hyper_out_dir / "logs",
            tags=[model_name, data_name, trainer_name, "train", "normal_hyper", f"diffusion_comparison"],
        ),
        best_checkpoint_command = add_best_checkpoint_command(
            dir_path=normal_train_hyper_out_dir / "ckpts",
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
            normal_train_hyper_out_dir / "confirm.txt"
        ),
    resources:
        constraint="COMPUTE_MODEL_RTX_3090_25G",  # More VRAM needed to materialize hyper edges
        runtime=2880,  # slower model...
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
        """
