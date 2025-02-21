import logging
from logging.handlers import RotatingFileHandler
import os
from contextvars import ContextVar

# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Context variable to store rank
rank_var = ContextVar('rank', default=None)

def set_rank(rank):
    """Set the rank for the current context"""
    rank_var.set(rank)

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RankAwareFormatter(logging.Formatter):
    """Custom formatter that adds rank information"""
    def format(self, record):
        # Get rank
        rank = rank_var.get()
        rank_str = f"rank{rank}" if rank is not None else "unranked"
        
        # Add to record
        record.processThread = f"[{rank_str}]"
        return super().format(record)

# Format for the logs
formatter = RankAwareFormatter('%(asctime)s - %(processThread)s - %(levelname)s - %(message)s')

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