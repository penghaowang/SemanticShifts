import os
import logging
from logging.handlers import RotatingFileHandler
import sys

# Define TRACE level (more detailed than DEBUG)
TRACE = 5
logging.addLevelName(TRACE, "TRACE")

def setup_logger(name, log_file=None, level=logging.INFO):
    """Set up and return a logger with the specified name and level
    
    Args:
        name: Logger name
        log_file: Path to the log file, if None, output only to console
        level: Level of the logger
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers to avoid duplicate configuration
    if logger.handlers:
        logger.handlers = []
    
    # Create console handler, showing only INFO level and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)
    
    # If log file path is provided, create file handler
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Configure file handler, record DEBUG level and above, limit size to 5MB, keep up to 3 backups
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
        file_handler.setLevel(logging.DEBUG)  # Record all levels in the file
        file_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
    
    return logger

# Add trace method to Logger class
def trace(self, message, *args, **kwargs):
    """
    Log a message with severity TRACE
    """
    if self.isEnabledFor(TRACE):
        self.log(TRACE, message, *args, **kwargs)

# Add the trace method to the Logger class
logging.Logger.trace = trace

# Global default logger
def get_default_logger():
    """Get the default global logger"""
    return setup_logger('default_logger', 'logs/application.log') 