import logging
from logging.handlers import RotatingFileHandler


LOG_LEVELS = {'__name__': logging.DEBUG,  # The logger used we use.
              # 'httpx': logging.WARNING
              }

for id, level in LOG_LEVELS.items():
    logging.getLogger(id).setLevel(level)

file_handler = RotatingFileHandler(f"log.log", maxBytes=500_000, backupCount=10)  # ~4K lines per logfile
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s-%(funcName)s:%(lineno)d] %(message)s",
        datefmt='%m-%dT%H:%M:%S')
)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(
    logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s-%(funcName)s:%(lineno)d] %(message)s",
        datefmt='%H:%M:%S')
)
logging.basicConfig(handlers=[stream_handler, file_handler])
LOGGER = logging.getLogger('__name__')
