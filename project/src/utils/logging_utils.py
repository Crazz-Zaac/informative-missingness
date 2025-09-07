from loguru import logger
from pathlib import Path
from typing import Union, List
from src.config.schemas import LoggingConfig, ModelTypeEnum
import sys

DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

def setup_logging(config: LoggingConfig, model_type: Union[str, ModelTypeEnum]):
    """Set up logging with proper path handling and Loguru format"""
    # Ensure log directory exists
    config.log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model type string
    if isinstance(model_type, ModelTypeEnum):
        model_str = model_type.value
    else:
        model_str = str(model_type)

    
    # Configure log file
    log_file = config.get_log_filepath(model_str)
    
    # Remove default logger
    logger.remove()
    
    # Add file handler with Loguru format (not Python logging format)
    logger.add(
        log_file,
        level=config.log_level.value.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",  # Loguru format
        rotation="10 MB",
        retention="10 days",
        enqueue=True,  # Thread-safe logging
        serialize=False
    )
    
    # Add console handler with colored format
    logger.add(
        sys.stdout,
        level=config.log_level.value.upper(),
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
        colorize=True
    )
    
    logger.info(f"Logging initialized for {model_str} at {log_file}")
    return log_file