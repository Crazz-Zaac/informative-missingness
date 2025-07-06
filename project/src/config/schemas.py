from pydantic import BaseModel, Field, model_validator
from typing import List, Literal, Optional
from pathlib import Path
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Optional, List, Dict, Literal, Union, Any


class ModelTypeEnum(str, Enum):
    RF = "RandomForest"
    GRU = "GRU"
    GRUD = "GRU_D"


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


class RandomForestGridSearchParams(BaseModel):
    n_estimators: List[int]
    max_depth: List[int]
    min_samples_split: List[int]
    class_weight: Optional[List[Union[None, dict[int, float]]]]


class RandomForestFixedParams(BaseModel):
    n_estimators: int
    max_depth: int
    random_state: int
    min_samples_split: int
    class_weight: Union[str, dict[int, float]]
    min_samples_leaf: Optional[int] = 1  # default if not specified


class RandomForestHyperParams(BaseModel):
    fixed_params: RandomForestFixedParams
    grid_search_params: Optional[RandomForestGridSearchParams] = None


# Model configurations for different models
class HyperParams(BaseModel):
    fixed_params: Dict[str, Any]
    grid_search_params: Dict[str, Any]

class ModelHyperParams(BaseModel):
    RandomForest: Optional[HyperParams] = None
    LogisticRegression: Optional[HyperParams] = None
    # more models to be added here later

# A dictionary to map model types to their hyperparameters
class ModelConfig(BaseModel):
    model_type: ModelTypeEnum
    hyperparameters: ModelHyperParams



# Tabular data configuration for the experiment
class TabularDataConfig(BaseModel):
    data_path: str
    window_size: int
    feature_type: Literal["numeric", "categorical"]
    aggregation_window_size: int = Field(
        2, gt=0, lt=25, description="Size of the aggregation window in days"
    )
    training_feature: str
    age_threshold: int
    insurance_type: str

    @model_validator(mode="before")
    def validate_training_feature(cls, values):
        if values.get("training_feature") not in [
            "target",
            "gender",
            "anchor_age",
            "race",
        ]:
            raise ValueError(
                "training_feature must be one of ['target', 'gender', 'anchor_age', 'race']"
            )
        if values.get("training_feature") == "anchor_age":
            if (
                values.get("age_threshold", 0) < 18
                or values.get("age_threshold", 0) >= 91
            ):
                raise ValueError(
                    "age_threshold must be between 18 and 90 for anchor_age feature"
                )
        return values


class TemporalDataConfig(BaseModel):
    data_path: str


class DataConfig(BaseModel):
    test_size: float
    tabular: TabularDataConfig
    temporal: TemporalDataConfig


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
    dataset_dir: Path = Path(__file__).resolve().parents[2] / "dataset"
    preprocessed_tabular_data_dir: Path = Path(__file__).resolve().parents[2] / "dataset" / "processed_tabular"
    preprocessed_temporal_data_dir: Path = Path(__file__).resolve().parents[2] / "dataset" / "processed_temporal"
    raw_data_dir: Path = Path(__file__).resolve().parents[2] / "dataset" / "raw"
    temporary_data_dir: Path = Path(__file__).resolve().parents[2] / "dataset" / "temp"
    logging_dir: Path = Path(__file__).resolve().parents[2] / "logs"
    plots_dir: Path = Path(__file__).resolve().parents[2] / "plots"
    
    experiment_name: str = Field(
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
    training: TrainingConfig
    logging: LoggingConfig
    evaluation: EvaluationConfig
    save_best_model: bool = True
    save_model_every: Optional[int] = 1
    best_model_path: Optional[Path] = None

    def create_dirs(self):
        """Create directories for the experiment."""
        self.preprocessed_temporal_data_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_tabular_data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.temporary_data_dir.mkdir(parents=True, exist_ok=True)
        self.logging_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
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
