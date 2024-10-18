import logging
from .config import Config

def setup_logging(config: Config):
    logging.basicConfig(
        level=config.logging.get("level", "INFO"),
        format=config.logging.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging is set up.")
