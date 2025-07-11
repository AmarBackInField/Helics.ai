"""
Logging utility for the chase communication module.
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[91m',      # Bright Red
        'CRITICAL': '\033[95m',   # Magenta
    }
    RESET = '\033[0m'  # Reset color
    
    def format(self, record):
        # Get the color for this log level
        color = self.COLORS.get(record.levelname, self.RESET)
        
        # Format the message with color
        colored_levelname = f"{color}{record.levelname}{self.RESET}"
        
        # Create a copy of the record to avoid modifying the original
        record_copy = logging.makeLogRecord(record.__dict__)
        record_copy.levelname = colored_levelname
        
        return super().format(record_copy)

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logger = logging.getLogger("chase_communication")
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler
file_handler = RotatingFileHandler(
    os.path.join(logs_dir, "chase_communication.log"),
    maxBytes=10485760,  # 10MB
    backupCount=5
)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Prevent propagation to root logger
logger.propagate = False

# Explicitly expose logging methods
def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs)

def critical(msg, *args, **kwargs):
    logger.critical(msg, *args, **kwargs)

# Export logger and logging methods
__all__ = ["logger", "debug", "info", "warning", "error", "critical"] 