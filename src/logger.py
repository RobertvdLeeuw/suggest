import logging
import logging.handlers
import multiprocessing
import os
from pathlib import Path

NOISY_LOGGERS = ["sqlalchemy.engine",
                 "sqlalchemy.dialects", 
                 "sqlalchemy.pool", 
                 "sqlalchemy.orm", 
                 "httpcore", 
                 "httpx", 
                 "apscheduler", 
                 "h5py", 
                 "pylast", 
                 "spotipy", 
                 "musicbrainz", 
                 "spotdl", 
                 "huggingface", 
                 "urllib3.connectionpool"]

class NumbaFilter(logging.Filter):
    def filter(self, record):
        return "numba.core" not in record.name

class MusicBrainzFilter(logging.Filter):
    def filter(self, record):
        return "musicbrainzngs" not in record.name or "parse_attributes" not in record.funcName

class InfoAndAboveNoisyFilter(logging.Filter):
    def filter(self, record):
        if not any(logger in record.name for logger in NOISY_LOGGERS): return True
        return record.levelno >= logging.INFO

class WarningAndAboveNoisyFilter(logging.Filter):
    def filter(self, record):
        if not any(logger in record.name for logger in NOISY_LOGGERS): return True
        return record.levelno >= logging.WARNING

def setup_multiprocess_logging(log_path: str = None, console_level=logging.INFO):
    """
    One function to set up logging for multiprocessing.
    Call this ONCE in your main process.
    """
    if log_path is None:
        log_folder = "test_logs" if os.getenv("TEST_MODE") else "logs"
        log_path = f"{log_folder}/log.log"
    
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        pass
    
    # Create the multiprocessing-safe file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10*1024*1024, backupCount=5
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level) 
    
    # Set formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s-%(funcName)s:%(lineno)d] %(message)s",
        datefmt='%H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    file_handler.addFilter(NumbaFilter())
    file_handler.addFilter(MusicBrainzFilter())
    file_handler.addFilter(InfoAndAboveNoisyFilter())

    console_handler.setFormatter(formatter)
    console_handler.addFilter(NumbaFilter())
    console_handler.addFilter(MusicBrainzFilter())
    console_handler.addFilter(WarningAndAboveNoisyFilter())
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    level_to_str = {logging.DEBUG: "debug",
                    logging.INFO: "info",
                    logging.WARNING: "warning",
                    logging.ERROR: "error",
                    }

    logging.info(f"Logging initialized (PID: {os.getpid()}) " \
                 f"(console level: {level_to_str[console_level]})")
    print(f"Logging initialized (PID: {os.getpid()}) " \
          f"(console level: {level_to_str[console_level]})")
