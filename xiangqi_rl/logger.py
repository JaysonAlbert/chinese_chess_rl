import logging
from logging.handlers import RotatingFileHandler
import os
import multiprocessing
import threading

# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ThreadProcessFormatter(logging.Formatter):
    """Custom formatter that adds thread and process information"""
    def format(self, record):
        # Get process name/id
        if hasattr(record, 'processName'):
            process_name = record.processName
        else:
            process_name = multiprocessing.current_process().name
            
        # Get thread name/id
        thread_name = threading.current_thread().name
        
        # Add to record
        record.processThread = f"[{process_name}|{thread_name}]"
        return super().format(record)

# Format for the logs
formatter = ThreadProcessFormatter('%(asctime)s - %(processThread)s - %(levelname)s - %(message)s')

# File handler (with rotation)
file_handler = RotatingFileHandler(
    os.path.join(log_dir, 'xiangqi.log'),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Add both handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Set logging level for other loggers
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)