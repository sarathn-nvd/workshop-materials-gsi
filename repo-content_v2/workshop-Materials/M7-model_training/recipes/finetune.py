# Thin launcher for NeMo AutoModel supervised fine-tuning.
# Copied verbatim from the production pipeline
# (gsi-training/7.run_sft/4.run_sft/finetune.py) -- identical to pretrain.py;
# the difference between CPT and SFT lives entirely in the --config recipe YAML
# (MegatronPretraining dataset vs ChatDataset). Runs INSIDE the
# nvcr.io/nvidia/nemo-automodel:26.04 container.
from __future__ import annotations

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction


def main(default_config_path="recipe.yaml"):
    cfg = parse_args_and_load_config(default_config_path)
    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
