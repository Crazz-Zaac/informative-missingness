from loguru import logger
from pathlib import Path
from src.config.schemas import LoggingConfig
import sys

DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

def setup_logging(config: LoggingConfig) -> Path:
    """Set up loguru-based logging using configuration."""
    log_dir = config.log_dir or DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{config.expirement_id}.log"
    
    logger.remove()  # Remove default handler

    # Add file logging
    logger.add(
        log_file,
        level=config.log_level.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
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

    logger.info(f"Loguru logging setup complete. Log file: {log_file}")
    return log_dir
