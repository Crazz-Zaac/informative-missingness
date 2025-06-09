from pydantic import BaseModel, Field, model_validator
from typing import List, Literal, Optional
from pathlib import Path
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Optional, List, Union, Literal


class ModelTypeEnum(str, Enum):
    RF = "RandomForestClassifier"
    GRU = "GRU"
    GRUD = "GRU-D"


class MetricsEnum(str, Enum):
    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"


class LoggingLevelEnum(str, Enum):
    INFO = "info"
    DEBUG = "debug"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RandomForestHyperParams(BaseModel):
    n_estimators: int
    max_depth: int
    random_state: int
    class_weight: str
    min_samples_split: int
    min_samples_leaf: int


class GRUHyperParams(BaseModel):
    hidden_size: int
    num_layers: int
    dropout: float


class TabularDataConfig(BaseModel):
    data_path: str
    window_size: int
    feature_type: Literal["numeric", "categorical"] = "numeric"
    aggregation_window_size: int = Field(
        2, gt=0, lt=25, description="Size of the aggregation window in days"
    )


class TemporalDataConfig(BaseModel):
    data_path: str


class DataConfig(BaseModel):
    test_size: float = 0.2
    tabular: TabularDataConfig
    temporal: TemporalDataConfig


class ModelConfig(BaseModel):
    model_type: ModelTypeEnum
    hyperparameters: Union[RandomForestHyperParams, GRUHyperParams, dict]


class TrainingConfig(BaseModel):
    max_epochs: int = Field(10, gt=0, description="Maximum number of training epochs")
    learning_rate: float = Field(
        0.001, gt=0.0, description="Learning rate for the optimizer"
    )


class LoggingConfig(BaseModel):
    log_path: Path = Path(__file__).resolve().parents[2] / "logs"
    log_dir: Path = Field(log_path, description="Directory for log files")
    expirement_id: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    log_level: LoggingLevelEnum
    log_format: str = "%(asctime)s - %(levelname)s - %(message)s"


class EvaluationConfig(BaseModel):
    metrics: List[MetricsEnum]


class ExperimentConfig(BaseModel):
    # should match with the one in the config.yml file
    expirement_name: str = Field(
        "default_experiment", description="Name of the experiment"
    )
    experiment_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "outputs"
        / "experiments"
    )
    experiment_id: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    data: DataConfig
    model: ModelConfig
    random_state: int = 42
    training: TrainingConfig
    logging: LoggingConfig
    evaluation: EvaluationConfig
    save_best_model: bool = True
    save_model_every: Optional[int] = 1
    best_model_path: Optional[Path] = None

    def create_dirs(self):
        """Create directories for the experiment."""
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "results").mkdir(parents=True, exist_ok=True)

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
