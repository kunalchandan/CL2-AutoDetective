# Logging imports
import logging
from logging.handlers import QueueHandler
from queue import Queue
from logging import Handler

LOGGING_LEVEL = 0


def set_queue_logger(logging_name: str, log_queue: Queue) -> logging.Logger:
    # Logging Settings
    queue_handler = QueueHandler(log_queue)
    logger = logging.getLogger(logging_name)
    logger.setLevel(LOGGING_LEVEL)
    logger.addHandler(queue_handler)
    return logger
