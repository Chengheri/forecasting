"""Utility decorators for function timing and logging."""

import time
from functools import wraps
from datetime import datetime, timedelta
from backend.app.utils.logger import Logger

# Initialize logger
logger = Logger()

def log_execution_time(func):
    """
    Decorator that logs the execution time of a function in a human-readable format.
    
    Args:
        func: The function to be decorated
        
    Returns:
        The decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start timing
        start_time = time.time()
        
        # Execute the function
        result = func(*args, **kwargs)
        
        # Calculate duration
        end_time = time.time()
        duration = end_time - start_time
        
        # Convert to human readable format
        duration_timedelta = timedelta(seconds=duration)
        hours = duration_timedelta.seconds // 3600
        minutes = (duration_timedelta.seconds % 3600) // 60
        seconds = duration_timedelta.seconds % 60
        milliseconds = int(duration_timedelta.microseconds / 1000)
        
        # Create human readable duration string
        duration_str = ""
        if hours > 0:
            duration_str += f"{hours}h "
        if minutes > 0:
            duration_str += f"{minutes}m "
        if seconds > 0:
            duration_str += f"{seconds}s "
        if milliseconds > 0:
            duration_str += f"{milliseconds}ms"
        
        # Log the execution time
        logger.info(f"Function '{func.__name__}' executed in {duration_str.strip()}")
        
        return result
    
    return wrapper 