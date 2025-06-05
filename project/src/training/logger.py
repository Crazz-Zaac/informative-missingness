import logging
from pathlib import Path
from src.config.schemas import ExperimentConfig, LoggingConfig
from src.utils.logging_utils import setup_logging  # assuming this sets up a log file and returns the path
import datetime

class ExperimentLogger:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        log_dir = Path(config.logging.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct unique log file path
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = log_dir / f"{config.experiment_id}_{timestamp}.log"

        # Setting up the actual logger
        self.logger = logging.getLogger(f"experiment_logger_{config.experiment_id}")
        self.logger.setLevel(config.logging.log_level.upper())

        # Avoiding duplicate handlers
        if not self.logger.handlers:
            file_handler = logging.FileHandler(self.log_path)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log_metrics(self, metrics: dict):
        """Log multiple metrics."""
        for name, value in metrics.items():
            self.logger.info(f"Metric - {name}: {value}")

    def log_parameters(self, params: dict):
        """Log model parameters."""
        for key, value in params.items():
            self.logger.info(f"Parameter - {key}: {value}")

    def finish(self):
        """Finalize logging and clean up."""
        self.logger.info("Experiment completed.")
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
