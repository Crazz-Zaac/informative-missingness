import concurrent.futures
from pathlib import Path
import yaml
import os
from loguru import logger
from src.training import TRAINER_REGISTRY
from src.config.schemas import ExperimentConfig, ModelTypeEnum
from src.utils.logging_utils import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yml"


def train_model(model_name: str, config: ExperimentConfig):
    model_type = ModelTypeEnum(model_name)  # Convert to enum for safety
    trainer_cls = TRAINER_REGISTRY[model_type.value.lower()]

    # Get the correct hyperparameters using attribute access
    hyperparams = getattr(config.model.hyperparameters, model_type.value)

    trainer = trainer_cls(config=config)
    trainer.run_training()


def main():
    with open(CONFIG_PATH, "r") as file:
        config_data = yaml.safe_load(file)
    config = ExperimentConfig(**config_data)

    setup_logging(config.logging, model_type=config.model.model_type)

    # Limit workers if needed
    max_workers = min(len(config.model.model_type), os.cpu_count() or 1)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(train_model, model_name, config): model_name
            for model_name in config.model.model_type
        }

        for future in concurrent.futures.as_completed(futures):
            model_name = futures[future]
            try:
                future.result()  # Will raise exceptions if any occurred
            except Exception as e:
                logger.error(f"Model {model_name} failed: {str(e)}")


if __name__ == "__main__":
    main()
