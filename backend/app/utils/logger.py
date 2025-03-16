import logging
import os
import inspect
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from typing import Optional, Dict, Any

class ModuleNameFilter(logging.Filter):
    """Filter that adds module name to log records."""
    def filter(self, record):
        frame = inspect.currentframe()
        try:
            # Walk up the call stack to find the actual caller
            f = frame
            while f:
                # Skip frames from logger.py and decorators.py
                if (f.f_code.co_filename.endswith('logger.py') or 
                    f.f_code.co_filename.endswith('decorators.py')):
                    f = f.f_back
                    continue
                
                # Found a frame that's not from logger or decorators
                if f.f_code.co_filename:
                    record.modulename = os.path.splitext(os.path.basename(f.f_code.co_filename))[0]
                    break
                f = f.f_back
            
            # If we couldn't find a suitable frame, use the record's pathname
            if not f or not hasattr(record, 'modulename'):
                record.modulename = os.path.splitext(os.path.basename(record.pathname))[0]
        finally:
            if frame:
                del frame  # Clean up circular reference
        return True

class Logger:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance
    
    def _initialize_logger(self):
        # Create logs directory with absolute path
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        log_dir = os.path.join(project_root, 'data', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Create root logger
        self.logger = logging.getLogger("forecasting")
        self.logger.setLevel(logging.INFO)
        
        # Create module name filter
        module_filter = ModuleNameFilter()
        
        # Create formatters with module name
        formatter = logging.Formatter(
            '%(asctime)s - [%(modulename)s] - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(module_filter)
        
        # File handler with daily rotation
        current_date = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(log_dir, f"forecasting_{current_date}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(module_filter)
        
        # Remove existing handlers if any
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False
    
    def _log(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None):
        """Internal method to handle logging with module name."""
        if extra:
            # Format extra fields into the message
            extra_str = ' '.join(f"{k}={v}" for k, v in extra.items())
            message = f"{message} | {extra_str}"
        
        # Get the logging method
        log_method = getattr(self.logger, level)
        # Log the message
        log_method(message)
    
    def info(self, message: str):
        """Log an info message."""
        self._log('info', message)
    
    def error(self, message: str):
        """Log an error message."""
        self._log('error', message)
    
    def warning(self, message: str):
        """Log a warning message."""
        self._log('warning', message)
    
    def debug(self, message: str):
        """Log a debug message."""
        self._log('debug', message)
    
    def critical(self, message: str):
        """Log a critical message."""
        self._log('critical', message) 