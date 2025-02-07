import logging
from logging.handlers import RotatingFileHandler
import os

# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Format for the logs
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

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