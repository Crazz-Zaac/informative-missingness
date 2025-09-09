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
    # Load and validate config
    with open(CONFIG_PATH, "r") as file:
        config_data = yaml.safe_load(file)
    
    # Explicitly set project root
    config_data["project_root"] = PROJECT_ROOT
    config = ExperimentConfig(**config_data)
    
    # Create all directories first
    config.create_dirs()
    
    # Setup base logging (before parallel execution)
    base_log_file = setup_logging(config.logging, model_type="experiment")
    logger.info(f"Experiment started (main log: {Path(base_log_file).stem})")
    logger.info(f"Project root: {Path(PROJECT_ROOT).stem}")
    logger.info(f"Experiment directory: {Path(config.experiment_dir).stem}")

    # Prepare models (validate before parallel execution)
    model_types = config.model.model_type
    if isinstance(model_types, str):
        model_types = [model_types]
    
    logger.info(f"Training models: {', '.join(model_types)}")
    
    for model_name in model_types:
        try:
            train_model(model_name, config)
            logger.success(f"{model_name} completed successfully")
        except Exception as e:
            logger.error(f"{model_name} failed: {str(e)}")

if __name__ == "__main__":
    main()