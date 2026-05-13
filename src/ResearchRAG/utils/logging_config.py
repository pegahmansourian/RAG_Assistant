import logging
from datetime import datetime
from pathlib import Path

from ResearchRAG.config import LOG_DIR

LOG_DIR.mkdir(exist_ok=True)


def setup_logging():
    root_logger = logging.getLogger()

    if root_logger.handlers:
        return

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    log_file = LOG_DIR / f"{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger(__name__)

    logger.info(
        "Logging initialized: %s",
        log_file.name
    )

    return log_file