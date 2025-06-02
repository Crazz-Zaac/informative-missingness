from pydantic import BaseModel, Field, model_validator
from typing import List, Literal, Optional
from pathlib import Path
from datetime import datetime
from pathlib import Path


class DataConfig(BaseModel):
    window_size: int = Field(
        7, description="Size of the sliding window for time series data"
    )
    batch_size: int = Field(32, gt=0)
    preprocessing_cache: str = Field(
        "../dataset/preprocessed/", description="Path to cache preprocessed data"
    )


class ModelConfig(BaseModel):
    model_type: Literal["rf", "gru", "grud"] = Field("rf")
    input_size: int = Field(224)
    dropout_rate: float = Field(0.5, ge=0.0, le=1.0)


class TrainingConfig(BaseModel):
    max_epochs: int = Field(10, gt=0, description="Maximum number of training epochs")
    learning_rate: float = Field(
        0.001, gt=0.0, description="Learning rate for the optimizer"
    )


class LoggingConfig(BaseModel):
    log_dir = Path(__file__).resolve().parents[2] / "logs"
    log_dir: Path = Field(log_dir, description="Directory for log files")
    expirement_id: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    log_level: Literal["info", "debug", "warning"] = Field("info")


class EvaluationConfig(BaseModel):
    metrics: List[Literal["accuracy", "f1", "precision", "recall"]] = Field(
        ["accuracy"]
    )


class ExperimentConfig(BaseModel):
    experiment_name: str = Field("default_experiment")
    experiment_dir: Path = Field(Path("../outputs/experiments"))
    experiment_id: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig
    evaluation: EvaluationConfig
    save_best_model: bool = True
    save_model_every: Optional[int] = 1
    best_model_path: Optional[Path] = None

    @model_validator(mode="before")
    def set_defaults(cls, values):
        exp_dir = Path(values.get("experiment_dir", "../outputs/experiments"))
        exp_id = values.get("experiment_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        values["experiment_dir"] = exp_dir
        values["experiment_id"] = exp_id
        if not exp_dir.exists():
            exp_dir.mkdir(parents=True, exist_ok=True)
        if values.get("save_best_model", True):
            values["best_model_path"] = exp_dir / f"{exp_id}.pth"
        return values

    def create_dirs(self):
        """Create directories for the experiment."""
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "results").mkdir(parents=True, exist_ok=True)
