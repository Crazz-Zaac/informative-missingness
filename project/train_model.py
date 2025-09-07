import argparse
import concurrent.futures
from pathlib import Path
import yaml
import os
from loguru import logger
from src.training import TRAINER_REGISTRY
from src.config.schemas import ExperimentConfig, ModelTypeEnum
from src.utils.logging_utils import setup_logging

# Define project root explicitly
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yml"


def train_model(model_name: str, config: ExperimentConfig):
    try:
        model_type = ModelTypeEnum(model_name)
        trainer_cls = TRAINER_REGISTRY[model_type.value.lower()]

        # Setup per-model logging
        model_log_file = setup_logging(config.logging, model_type=model_type)
        logger.info(f"Starting training for {model_type.value} (log: {model_log_file})")

        trainer = trainer_cls(config=config)
        return trainer.run_training()
    except Exception as e:
        logger.error(f"Model {model_name} training failed: {str(e)}")
        raise


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to train (if not provided, train all from config)",
    )
    args = parser.parse_args()

    # Load and validate config
    with open(CONFIG_PATH, "r") as file:
        config_data = yaml.safe_load(file)

    # Explicitly set project root
    config_data["project_root"] = PROJECT_ROOT
    config = ExperimentConfig(**config_data)

    # Create all directories first
    config.create_dirs()

    # Setup base logging
    base_log_file = setup_logging(config.logging, model_type=args.model)
    logger.info(f"Experiment started (main log: {base_log_file})")

    # Train a model
    if args.model:
        logger.info(f"Training single model: {args.model}")
        train_model(args.model, config)
        return
    


if __name__ == "__main__":
    main()
