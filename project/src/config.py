from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from pathlib import Path
from enum import Enum


class DataConfig(BaseModel):
    batch_size: int = Field(32)
    num_workers: int = Field(4)
    shuffle: bool = Field(True)
    validation_split: float = Field(0.2, ge=0.0, le=1.0)


class ModelConfig(BaseModel):
    architecture: Literal["rf", "gru", "grud"] = Field("rf")
    num_classes: int = Field(10)
    input_size: int = Field(224)
    dropout_rate: float = Field(0.5, ge=0.0, le=1.0)


class TrainingConfig(BaseModel):
    epochs: int = Field(10)
    learning_rate: float = Field(0.001, gt=0.0)
    weight_decay: float = Field(0.0001, ge=0.0)
    early_stopping_patience: int = Field(5)
    save_best_model: bool = Field(True)
    if save_best_model:
        experiment_name: str = Field("best_model")
        save_model_every: int = Field(1)
        best_model_path: Path = Field(Path(f"outputs/experiments/{experiment_name}.pth"))


class LoggingConfig(BaseModel):
    log_dir: Path = Field(Path("outputs/experiments/logs"))
    log_level: Literal["info", "debug", "warning"] = Field("info")


class EvaluationConfig(BaseModel):
    metrics: List[Literal["accuracy", "f1", "precision", "recall"]] = Field(
        ["accuracy"]
    )

class ExperimentConfig(BaseModel):
    experiment_name: str = Field("default_experiment")
    experiment_dir: Path = Field(Path("outputs/experiments"))
    data: DataConfig = Field(DataConfig())
    model: ModelConfig = Field(ModelConfig())
    training: TrainingConfig = Field(TrainingConfig())
    logging: LoggingConfig = Field(LoggingConfig())
    evaluation: EvaluationConfig = Field(EvaluationConfig())

    def __init__(self, **data):
        super().__init__(**data)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)