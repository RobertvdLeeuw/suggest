import logging
from logging.handlers import TimedRotatingFileHandler
import sqlalchemy
import os
import threading
import time
from collections import deque
from pathlib import Path

# Set SQLAlchemy engine logging to WARNING to reduce output
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.dialects').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.orm').setLevel(logging.WARNING)

class BufferedLogger:
    """
    A logger that buffers messages and flushes them periodically
    This reduces file I/O conflicts without needing a separate thread
    """
    def __init__(self, log_path: str, buffer_size: int = 10, flush_interval: float = 2.0):
        self.log_path = log_path
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        # Thread-safe deque for buffering
        self.buffer = deque()
        self.lock = threading.Lock()
        self.last_flush = time.time()
        
        # Create the actual logger
        self.logger = self._create_logger()
    
    def _create_logger(self):
        """Create the underlying logger"""
        # Ensure directory exists
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create logger with only console handler initially
        logger = logging.getLogger(f"__name___{os.getpid()}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Only add stream handler - file writing is handled by buffer
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s [%(name)s-%(funcName)s:%(lineno)d] %(message)s",
                datefmt='%H:%M:%S')
        )
        logger.addHandler(stream_handler)
        
        return logger
    
    def _should_flush(self):
        """Check if we should flush the buffer"""
        return (len(self.buffer) >= self.buffer_size or 
                time.time() - self.last_flush > self.flush_interval)
    
    def _flush_buffer(self):
        """Flush buffered messages to file"""
        if not self.buffer:
            return
        
        try:
            # Write all buffered messages at once
            with open(self.log_path, 'a', encoding='utf-8') as f:
                while self.buffer:
                    f.write(self.buffer.popleft() + '\n')
                f.flush()
                os.fsync(f.fileno())  # Force OS write
            
            self.last_flush = time.time()
        except Exception as e:
            # If file write fails, at least we have console output
            print(f"Failed to write to log file: {e}")
    
    def _log_message(self, level: str, message: str):
        """Internal method to handle logging"""
        import inspect
        
        # Get caller info
        frame = inspect.currentframe().f_back.f_back
        func_name = frame.f_code.co_name
        line_no = frame.f_lineno
        
        # Log to console immediately
        getattr(self.logger, level.lower())(message)
        
        # Buffer for file writing
        timestamp = time.strftime('%m-%dT%H:%M:%S')
        log_entry = f"[{timestamp}] {level.upper()} [__name__-{func_name}:{line_no}] {message}"
        
        with self.lock:
            self.buffer.append(log_entry)
            
            # Flush if necessary
            if self._should_flush():
                self._flush_buffer()
    
    def debug(self, message: str):
        self._log_message('DEBUG', message)
    
    def info(self, message: str):
        self._log_message('INFO', message)
    
    def warning(self, message: str):
        self._log_message('WARNING', message)
    
    def error(self, message: str):
        self._log_message('ERROR', message)
    
    def flush(self):
        """Manually flush the buffer"""
        with self.lock:
            self._flush_buffer()
    
    def __del__(self):
        """Ensure buffer is flushed when object is destroyed"""
        try:
            self.flush()
        except:
            pass

def new_logger(log_path: str):
    """
    Create a buffered logger that reduces I/O conflicts
    """
    return BufferedLogger(log_path)

LOGGER = None
def get_logger():
    global LOGGER

    if LOGGER is None:
        log_folder = "test_logs" if os.getenv("TEST_MODE") else "logs"
        log_path = f"{log_folder}/log.log"
        
        LOGGER = new_logger(log_path)
        LOGGER.info(f"Logger initialized with path '{log_path}' (PID: {os.getpid()})")

    return LOGGER

# Optional: Ensure final flush on program exit
import atexit
def cleanup_logger():
    global LOGGER
    if LOGGER:
        LOGGER.flush()

atexit.register(cleanup_logger)
