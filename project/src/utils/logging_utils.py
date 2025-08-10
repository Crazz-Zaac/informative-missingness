from loguru import logger
from pathlib import Path
from typing import Union, List
from src.config.schemas import LoggingConfig, ModelTypeEnum
import sys

DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

def setup_logging(
    config: LoggingConfig,
    model_type: Union[ModelTypeEnum, str, List[Union[ModelTypeEnum, str]]]
) -> Path:
    """Set up loguru-based logging using configuration."""
    log_dir = config.log_dir or DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    # Normalize model_type into a string
    if isinstance(model_type, list):
        model_type_str = "_".join(
            mt.value if isinstance(mt, ModelTypeEnum) else str(mt)
            for mt in model_type
        )
    else:
        model_type_str = model_type.value if isinstance(model_type, ModelTypeEnum) else str(model_type)

    log_file = config.get_log_filepath(model_type=model_type_str)

    logger.remove()  # Remove default handler

    # File logging
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

    # Console logging
    logger.add(
        sys.stdout,
        level=config.log_level.value.upper(),
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )

    logger.info(f"Logging initialized for {model_type_str} at {log_file}")
    return log_dir
