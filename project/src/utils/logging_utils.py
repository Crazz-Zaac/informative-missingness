from src.config.schemas import LoggingConfig
import logging
from pathlib import Path
import os


DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

def setup_logging(config: LoggingConfig) -> Path:
    """Set up logging configuration."""
    log_dir: Path = config.log_dir if config.log_dir else DEFAULT_LOG_DIR
    log_file = log_dir / f"{config.expirement_id}.log"
    expirement_id = config.expirement_id
    log_level = config.log_level
    if not expirement_id:
        raise ValueError("Experiment ID must be provided in the logging configuration.")
    
    # clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=log_file,
        level=getattr(logging, config.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(console_handler)
    
    logging.info("Logging setup complete. Log file: %s", log_file)
    return log_dir
