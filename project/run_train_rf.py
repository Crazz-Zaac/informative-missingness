from pathlib import Path
from src.training.train_rf import RandomForestTrainer
from src.config.schemas import ExperimentConfig
from src.utils.logging_utils import setup_logging
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent  # project root directory
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yml"
if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as file:
        config_data = yaml.safe_load(file)
    config = ExperimentConfig(**config_data)

    # setup logging
    log_dir = setup_logging(config.logging)

    trainer = RandomForestTrainer.from_yaml(config_path=CONFIG_PATH)
    trained_model = trainer.run_training()
