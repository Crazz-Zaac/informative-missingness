from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from pathlib import Path
from enum import Enum
from datetime import datetime


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
    experiment_id: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig
    evaluation: EvaluationConfig
    save_best_model: bool = Field(True)
    if save_best_model:
        save_model_every: int = Field(1)
        # create a directory with experiment_id
        try:
            experiment_dir = Path(experiment_dir)
        except Exception as e:
            raise ValueError(
                f"Invalid experiment directory: {experiment_dir}. Error: {e}"
            )
        best_model_path: Path = Field(
            Path(f"{experiment_dir}/{experiment_id}.pth")
        )

    def __init__(self, **data):
        super().__init__(**data)
        # Create the experiment directory if it doesn't exist
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
