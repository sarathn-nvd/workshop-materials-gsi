# Thin launcher for NeMo AutoModel pre-training / fine-tuning.
# Copied verbatim from the production pipeline
# (gsi-training/3.cpt/2.run_cpt/run-v2/3.run_cpt/pretrain.py). Everything is
# driven by the --config recipe YAML; this file only parses it and runs the
# next-token-prediction training recipe. Runs INSIDE the
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
