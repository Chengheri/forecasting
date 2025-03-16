"""Base class providing logging functionality for all classes."""

from typing import Dict, Any, Optional
from .logger import Logger

class LoggedObject:
    """Base class that provides logging functionality.
    
    This class should be inherited by any class that needs logging capabilities.
    It automatically sets up a logger with the correct module name based on the class.
    """
    
    def __init__(self):
        """Initialize logging functionality."""
        self._logger = Logger()
        # Set the logger name to the actual class module and name
        self._logger.logger.name = f"{self.__class__.__module__}.{self.__class__.__name__}"
    
    def _log(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None):
        """Generic logging method that handles all log levels.
        
        Args:
            level: The log level ('info', 'error', 'warning', 'debug', 'critical')
            message: The message to log
            extra: Optional extra fields to include in the log
        """
        log_method = getattr(self._logger, level)
        log_method(message, extra=extra)
    
    def _log_info(self, message: str, **kwargs):
        """Log info message with proper module context."""
        self._log('info', message, kwargs)
    
    def _log_error(self, message: str, **kwargs):
        """Log error message with proper module context."""
        self._log('error', message, kwargs)
    
    def _log_warning(self, message: str, **kwargs):
        """Log warning message with proper module context."""
        self._log('warning', message, kwargs)
    
    def _log_debug(self, message: str, **kwargs):
        """Log debug message with proper module context."""
        self._log('debug', message, kwargs)
    
    def _log_critical(self, message: str, **kwargs):
        """Log critical message with proper module context."""
        self._log('critical', message, kwargs)
    
    @property
    def logger(self):
        """Get the logger instance.
        
        This property exists for backward compatibility but direct use
        of _log_* methods is preferred.
        """
        return self._logger 