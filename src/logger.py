import logging
import logging.handlers
import multiprocessing
import os
from pathlib import Path

def setup_multiprocess_logging(log_path: str = None):
    """
    One function to set up logging for multiprocessing.
    Call this ONCE in your main process.
    """
    if log_path is None:
        log_folder = "test_logs" if os.getenv("TEST_MODE") else "logs"
        log_path = f"{log_folder}/log.log"
    
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create the multiprocessing-safe file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10*1024*1024, backupCount=5
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    
    # Set formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s-%(funcName)s:%(lineno)d] %(message)s",
        datefmt='%H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Disable SQLAlchemy verbose logging
    for logger_name in ['sqlalchemy.engine', 'sqlalchemy.dialects', 
                       'sqlalchemy.pool', 'sqlalchemy.orm']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    logging.info(f"Logging initialized (PID: {os.getpid()})")
