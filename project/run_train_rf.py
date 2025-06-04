# run_rf_train.py

from pathlib import Path
from src.training.train_rf import RandomForestTrainer

PROJECT_ROOT = Path(__file__).resolve().parent  # Adjust based on your structure
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yml"
if __name__ == "__main__":
    config_path = Path(CONFIG_PATH)  # Adjust path if needed
    trainer = RandomForestTrainer.from_yaml(config_path)
    trained_model = trainer.run_training()
