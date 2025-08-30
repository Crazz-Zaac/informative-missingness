from pydantic import BaseModel, Field, model_validator
from typing import List, Literal, Optional
from pathlib import Path
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Optional, List, Dict, Literal, Union, Any


class ModelTypeEnum(str, Enum):
    RF = "RandomForest"
    GradBoost = "GradientBoosting"
    LR = "LogisticRegression"
    XGBoost = "XGBoost"
    CatBoost = "CatBoost"


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


# Random Forest model hyperparameters
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


# Gradient Boosting model hyperparameters
class GradBoostGridSearchParams(BaseModel):
    learning_rate: List[float]
    max_iter: List[int]
    max_depth: List[Optional[int]]


class GradBoostFixedParams(BaseModel):
    learning_rate: float
    max_iter: int
    max_depth: int
    class_weight: Union[str, dict[int, float]]


class GradBoostHyperParams(BaseModel):
    fixed_params: GradBoostFixedParams
    grid_search_params: Optional[GradBoostGridSearchParams] = None


# CatBoost model hyperparameters
class CatBoostGridSearchParams(BaseModel):
    learning_rate: List[float]
    depth: List[int]
    iterations: List[int]
    scale_pos_weight: List[float]
    objective: List[str]
    eval_metric: List[str]


class CatBoostFixedParams(BaseModel):
    learning_rate: float
    depth: int
    iterations: int
    scale_pos_weight: float
    objective: str
    eval_metric: str


class CatBoostHyperParams(BaseModel):
    fixed_params: CatBoostFixedParams
    grid_search_params: Optional[CatBoostGridSearchParams] = None


# Logistic Regression model hyperparameters
class LogisticRegressionFixedParams(BaseModel):
    penalty: str
    C: float
    solver: str


class LogisticRegressionGridSearchParams(BaseModel):
    penalty: List[str]
    C: List[float]
    solver: List[str]


class LogisticRegressionHyperParams(BaseModel):
    fixed_params: LogisticRegressionFixedParams
    grid_search_params: Optional[LogisticRegressionGridSearchParams] = None


# XGBoost model hyperparameters
class XGBoostFixedParams(BaseModel):
    n_estimators: int
    max_depth: int
    learning_rate: float
    scale_pos_weight: Optional[float]  # for imbalanced datasets
    objective: Optional[str]  # default objective for binary
    eval_metric: Optional[str]


class XGBoostGridSearchParams(BaseModel):
    n_estimators: List[int]
    max_depth: List[int]
    learning_rate: List[float]
    scale_pos_weight: Optional[List[float]]
    objective: Optional[List[str]]


class XGBoostHyperParams(BaseModel):
    fixed_params: XGBoostFixedParams
    grid_search_params: Optional[XGBoostGridSearchParams] = None


# Model configurations for different models
class HyperParams(BaseModel):
    fixed_params: Dict[str, Any]
    grid_search_params: Dict[str, Any]


class ModelHyperParams(BaseModel):
    RandomForest: Optional[HyperParams] = None
    GradientBoosting: Optional[HyperParams] = None
    LogisticRegression: Optional[HyperParams] = None
    XGBoost: Optional[HyperParams] = None
    CatBoost: Optional[HyperParams] = None


# A dictionary to map model types to their hyperparameters
class ModelConfig(BaseModel):
    model_type: Union[ModelTypeEnum, List[ModelTypeEnum]]
    hyperparameters: ModelHyperParams


# Tabular data configuration for the experiment
class TabularDataConfig(BaseModel):
    data_path: str
    training_data: str
    window_size: int
    feature_type: Literal["numeric", "categorical"]
    feature_combinations: Literal[
        "x", "m", "delta", "x_m", "x_delta", "m_delta", "x_m_delta"
    ]
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
    tabular: TabularDataConfig
    temporal: TemporalDataConfig


class LoggingConfig(BaseModel):
    log_dir: Path = Field(
        default_factory=lambda: Path("logs")
    )  # Relative to experiment_dir
    experiment_id: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    log_level: LoggingLevelEnum = LoggingLevelEnum.INFO
    log_format: str = "%(asctime)s - %(levelname)s - %(message)s"

    def get_log_filepath(self, model_type: str) -> Path:
        return self.log_dir / f"{model_type}_{self.experiment_id}.log"


class ExperimentConfig(BaseModel):
    # Get project root dynamically - this will be set from main script
    project_root: Optional[Path] = None

    # These will be computed based on project_root
    dataset_dir: Optional[Path] = None
    preprocessed_tabular_data_dir: Optional[Path] = None
    # preprocessed_temporal_data_dir: Optional[Path] = None
    raw_data_dir: Optional[Path] = None
    temporary_data_dir: Optional[Path] = None
    logging_dir: Optional[Path] = None
    plots_dir: Optional[Path] = None

    experiment_name: str = Field(
        "default_experiment", description="Name of the experiment"
    )
    experiment_dir: Optional[Path] = None
    experiment_id: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    data: DataConfig
    model: ModelConfig
    logging: LoggingConfig
    save_best_model: bool = True
    save_model_every: Optional[int] = 1
    best_model_path: Optional[Path] = None

    def create_dirs(self):
        """Create directories for the experiment."""
        # self.preprocessed_temporal_data_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_tabular_data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.temporary_data_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "results").mkdir(parents=True, exist_ok=True)

    @model_validator(mode="before")
    def set_defaults(cls, values):
        # Set project_root if not provided
        if values.get("project_root") is None:
            # Default fallback - try to find project root
            values["project_root"] = Path.cwd()

        project_root = Path(values["project_root"])

        # Set all directory paths relative to project root
        values["dataset_dir"] = project_root / "dataset"
        values["preprocessed_tabular_data_dir"] = (
            project_root / "dataset" / "preprocessed_tabular"
        )
        # values["preprocessed_temporal_data_dir"] = (
        #     project_root / "dataset" / "preprocessed_temporal"
        # )
        values["raw_data_dir"] = project_root / "dataset" / "raw"
        values["temporary_data_dir"] = project_root / "dataset" / "temp"

        exp_id = values.get("experiment_id")
        if exp_id is None:
            exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            values["experiment_id"] = exp_id

        # Set experiment directory within project
        exp_dir = project_root / "outputs" / "experiments" / exp_id
        values["experiment_dir"] = exp_dir

        # subdirectories
        logs_dir = exp_dir / "logs"
        models_dir = exp_dir / "models"
        results_dir = exp_dir / "results"

        # Ensure the experiment directory exists
        for dir_path in (logs_dir, models_dir, results_dir):
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)

        # Point logging and model paths to the correct directories
        logging_config = values.get("logging", {})
        logging_config["log_dir"] = logs_dir
        logging_config["experiment_id"] = exp_id
        values["logging"] = logging_config

        if values.get("save_best_model", True):
            values["best_model_path"] = models_dir / f"{exp_id}.pth"

        return values
