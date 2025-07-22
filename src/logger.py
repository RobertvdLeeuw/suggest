import logging
from logging.handlers import TimedRotatingFileHandler

import sqlalchemy
import os

# Set SQLAlchemy engine logging to WARNING to reduce output
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.dialects').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.orm').setLevel(logging.WARNING)

def new_logger(log_path: str):
    with open(log_path, 'w'): # TODO: OS remove all old logs, not just most recent.
        pass

    file_handler = TimedRotatingFileHandler(log_path, when='midnight', interval=1, backupCount=7)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s-%(funcName)s:%(lineno)d] %(message)s",
            datefmt='%m-%dT%H:%M:%S')
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s-%(funcName)s:%(lineno)d] %(message)s",
            datefmt='%H:%M:%S')
    )
    # logging.basicConfig(handlers=[stream_handler, file_handler])

    # LOG_LEVELS = {"__name__": logging.DEBUG,  # The logger used we use.
                  # 'sqlalchemy.engine': logging.WARNING,
                  # 'sqlalchemy.core': logging.WARNING,
                  # 'sqlalchemy.orm': logging.WARNING,
                  # 'sqlalchemy.dialects': logging.WARNING,
                  # 'sqlalchemy.pool': logging.WARNING,
                  # '': logging.WARNING
                  # }

    # for id, level in LOG_LEVELS.items():
        # logging.getLogger(id).setLevel(level)

    logger = logging.getLogger("__name__")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger

LOGGER = new_logger("logs/log.log")
