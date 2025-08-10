from loguru import logger
from pathlib import Path
from typing import Union, List
from src.config.schemas import LoggingConfig, ModelTypeEnum
import sys

DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

def setup_logging(config: LoggingConfig, model_type: Union[ModelTypeEnum, List[ModelTypeEnum]]) -> Path:
    """Set up loguru-based logging using configuration."""
    log_dir = config.log_dir or DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(model_type, list):
        model_type_str = "_".join([mt.value for mt in model_type])
    else:
        model_type_str = model_type.value

    log_file = config.get_log_filepath(model_type=model_type_str)

    logger.remove()  # Remove default handler

    # Add file logging
    logger.add(
        log_file,
        level=config.log_level.value.upper(),
        format=config.log_format,
        rotation="10 MB",
        retention="10 days",
        compression="zip",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    # Add console logging
    logger.add(
        sys.stdout,
        level=config.log_level.value.upper(),
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )

    logger.info(f"Logging initialized for {model_type} at {log_file}")
    return log_dir
