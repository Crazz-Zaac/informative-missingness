from pydantic import BaseModel
from loguru import logger
from pathlib import Path


class TabularPreprocessingConfig(BaseModel):
    raw_data_dir: Path
    training_data: str  # Name of the training data file
    preprocessed_data_dir: Path
    window_size: int
    aggregation_window_size: int  # aggregation by hours, e.g., 2 hours
    feature_combinations: (
        str  # e.g., ["x", "m", "delta", "x_m", "x_delta", "m_delta", "x_m_delta"]
    )
    feature_type: str
    training_feature: str
    age_threshold: int
    insurance_type: str

    @classmethod
    def from_defaults(
        cls,
        training_data: str,
        window_size: int,
        aggregation_window_size: int,
        feature_type: str,
        training_feature: str,
        feature_combinations: str,
        age_threshold: int,  # Default age threshold for filtering patients
        insurance_type: str,
    ) -> "TabularPreprocessingConfig":
        """Create a configuration instance with default values."""
        parent_dir = Path(__file__).parent.parent.parent
        return cls(
            raw_data_dir=parent_dir / "dataset" / "raw",
            training_data=training_data, 
            preprocessed_data_dir=parent_dir / "dataset" / "preprocessed_tabular",
            window_size=window_size,
            aggregation_window_size=aggregation_window_size,
            feature_combinations=feature_combinations,
            feature_type=feature_type,
            training_feature=training_feature,
            age_threshold=age_threshold,
            insurance_type=insurance_type,
        )

# class TemporalPreprocessingConfig(BaseModel):
#     raw_data_dir: Path
#     preprocessed_data_dir: Path
#     window_size: int
#     aggregation_window_size: int  # aggregation by hours, e.g., 2 hours
#     feature_combinations: (
#         str  # e.g., ["x", "m", "delta", "x_m", "x_delta", "m_delta", "x_m_delta"]
#     )
#     feature_type: str
#     training_feature: str
#     age_threshold: int
#     insurance_type: str

#     @classmethod
#     def from_defaults(
#         cls,
#         raw_data_dir: Path = Path("data/raw"),
#         preprocessed_data_dir: Path = Path("data/preprocessed"),
#         window_size: int = 24,
#         aggregation_window_size: int = 2,
#         feature_combinations: str = "x_m_delta",
#         feature_type: str = "temporal",
#         training_feature: str = "target",
#         age_threshold: int = 18,
#         insurance_type: str = "health",
#     ) -> "TemporalPreprocessingConfig":
#         """Create a configuration instance with default values."""
#         return cls(
#             raw_data_dir=raw_data_dir,
#             preprocessed_data_dir=preprocessed_data_dir,
#             window_size=window_size,
#             aggregation_window_size=aggregation_window_size,
#             feature_combinations=feature_combinations,
#             feature_type=feature_type,
#             training_feature=training_feature,
#             age_threshold=age_threshold,
#             insurance_type=insurance_type,
#         )