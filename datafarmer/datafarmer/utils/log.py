import logging

logger = logging.getLogger(__name__)

def setup_logger():
    """
    setup the logger with a stream handler and formatter
    """

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        log_format = "%(asctime)s | %(levelname)s: %(message)s"
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)

setup_logger()