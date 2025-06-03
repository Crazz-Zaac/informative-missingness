from pathlib import Path
from src.config.schemas import ExperimentConfig
import yaml


# the file is located in project/src/config/config.yaml
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def load_config(config_path: str) -> ExperimentConfig:
    """Load configuration from a YAML file or return default configuration."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")

    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)
    # convert the loaded dict to the ExperimentConfig model

    config = ExperimentConfig(**config_data)
    config.create_dirs()
    print(f"✅ Loaded config from: {config_path}")
    return config
