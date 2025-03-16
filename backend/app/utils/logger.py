import logging
import os
import inspect
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from typing import Optional, Dict, Any

class ModuleNameFilter(logging.Filter):
    """Filter that adds module name to log records."""
    def filter(self, record):
        if not hasattr(record, 'modulename'):
            # Find the actual caller's module
            calling_frame = self._get_calling_frame()
            if calling_frame:
                # Extract just the filename without extension
                record.modulename = os.path.splitext(os.path.basename(calling_frame.f_code.co_filename))[0]
            else:
                # Fallback to the record's pathname
                record.modulename = os.path.splitext(os.path.basename(record.pathname))[0]
                
        return True
        
    def _get_calling_frame(self):
        """Find the frame that actually called the logger (outside of logger.py and other internal modules)."""
        internal_modules = [
            'logger.py', 
            'decorators.py',
            'logging/__init__.py'
        ]
        
        # Start with the current frame and walk up the call stack
        frame = inspect.currentframe()
        try:
            # Skip this method's frame
            if frame:
                frame = frame.f_back
            else:
                return None
            
            # Walk up the stack
            while frame:
                filename = frame.f_code.co_filename
                # Skip frames from internal modules
                skip = False
                for internal_module in internal_modules:
                    if filename.endswith(internal_module):
                        skip = True
                        break
                        
                if not skip:
                    return frame
                    
                frame = frame.f_back
            
            # If we couldn't find a suitable frame
            return None
        finally:
            # Clean up circular references
            if frame:
                del frame

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
        if extra is None:
            extra = {}
            
        # Get the logging method
        log_method = getattr(self.logger, level)
        
        # Log the message
        log_method(message, extra=extra)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log an info message."""
        self._log('info', message, extra)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log an error message."""
        self._log('error', message, extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log a warning message."""
        self._log('warning', message, extra)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log a debug message."""
        self._log('debug', message, extra)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log a critical message."""
        self._log('critical', message, extra) 