import logging
from logging.handlers import RotatingFileHandler

import sqlalchemy

# Set SQLAlchemy engine logging to WARNING to reduce output
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.dialects').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.orm').setLevel(logging.WARNING)

LOG_PATH = "log.log"
with open(LOG_PATH, 'w'):
    pass

file_handler = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=3)
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

LOGGER = logging.getLogger("__name__")
LOGGER.setLevel(logging.DEBUG)
LOGGER.propagate = False
LOGGER.addHandler(stream_handler)
LOGGER.addHandler(file_handler)
