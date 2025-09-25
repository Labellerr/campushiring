from loguru import logger
from pathlib import Path


def configure_logging(log_dir: str | Path = "runs/logs", level: str = "INFO") -> None:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(log_path / "app.log", rotation="5 MB", retention=5, level=level)
    logger.add(lambda m: print(m, end=""), level=level)


__all__ = ["logger", "configure_logging"]
