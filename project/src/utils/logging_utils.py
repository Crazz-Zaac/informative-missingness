import logging
from pathlib import Path
import os


DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

def setup_logging(expirement_id: str, log_dir: Path, log_level: str = "info") -> None:
    """Set up logging configuration."""
    if not isinstance(log_dir, Path):
        raise ValueError("log_dir must be a Path object.")
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR
    
    # file logging
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{expirement_id}.log"
    
    # clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=log_file,
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(console_handler)
    
    logging.info("Logging setup complete. Log file: %s", log_file)
