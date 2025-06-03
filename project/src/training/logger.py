from src.utils.logging_utils import setup_logging, LoggingConfig
from src.config.schemas import ExperimentConfig
from pathlib import Path
import logging


class ExperimentLogger:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = setup_logging(
            LoggingConfig(
                expirement_id=config.experiment_id,
                log_dir=Path(config.logging.log_dir),
                log_level=config.logging.log_level,
            )
        )

    def log_metrics(self, name: str, value: float):
        """Log a single metric."""
        logging.info(f"{name}: {value}")

    def log_parameters(self, params: dict):
        """Log model parameters."""
        for key, value in params.items():
            logging.info(f"Parameter {key}: {value}")
