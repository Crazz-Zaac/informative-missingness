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
    logger.info(f"Experiment started (main log: {base_log_file})")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Experiment directory: {config.experiment_dir}")
    
    # Prepare models (validate before parallel execution)
    model_types = config.model.model_type
    if isinstance(model_types, str):
        model_types = [model_types]
    
    logger.info(f"Training models: {', '.join(model_types)}")
    
    # Limit workers based on resources
    max_workers = min(len(model_types), os.cpu_count() or 1)
    logger.info(f"Using {max_workers} parallel workers")
    
    # Run training
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(train_model, model_name, config): model_name
            for model_name in model_types
        }
        
        for future in concurrent.futures.as_completed(futures):
            model_name = futures[future]
            try:
                future.result()
                logger.success(f"{model_name} completed successfully")
            except Exception as e:
                logger.error(f"{model_name} failed: {str(e)}")

if __name__ == "__main__":
    main()