import argparse
from pathlib import Path
import yaml
from loguru import logger
from src.training import TRAINER_REGISTRY
from src.config.schemas import ExperimentConfig, ModelTypeEnum
from src.utils.logging_utils import setup_logging

# Define project root explicitly
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yml"

def train_single_model(model_name: str):
    # Load and validate config
    with open(CONFIG_PATH, "r") as file:
        config_data = yaml.safe_load(file)
    
    # Override model_type with the single model we want to train
    config_data["model"]["model_type"] = [model_name]
    config_data["project_root"] = PROJECT_ROOT
    
    config = ExperimentConfig(**config_data)
    config.create_dirs()
    
    # Setup logging
    log_file = setup_logging(config.logging, model_type=model_name)
    logger.info(f"Starting training for {model_name} (log: {log_file})")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Experiment directory: {config.experiment_dir}")
    
    # Train the model
    try:
        model_type = ModelTypeEnum(model_name)
        trainer_cls = TRAINER_REGISTRY[model_type.value.lower()]
        trainer = trainer_cls(config=config)
        result = trainer.run_training()
        logger.success(f"{model_name} completed successfully")
        return result
    except Exception as e:
        logger.error(f"{model_name} failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a single model')
    parser.add_argument('--model', type=str, required=True, 
                       help='Name of the model to train')
    args = parser.parse_args()
    
    train_single_model(args.model)