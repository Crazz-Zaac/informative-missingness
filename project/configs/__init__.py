from src.config.schemas import DataConfig, ModelConfig, TrainingConfig

import yaml

def load_configs() -> tuple[DataConfig, ModelConfig, TrainingConfig]:
    with open("configs/data_config.yaml", "r") as f:
        data_config = DataConfig(**yaml.safe_load(f))
    
    with open("configs/model_config.yaml", "r") as f:
        model_config = ModelConfig(**yaml.safe_load(f))
    
    with open("configs/training_config.yaml", "r") as f:
        training_config = TrainingConfig(**yaml.safe_load(f))
    
    return data_config, model_config, training_config