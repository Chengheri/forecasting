"""Base class for all pipeline implementations."""

from typing import Dict, Any
from ..utils.base import LoggedObject

class BasePipeline(LoggedObject):
    """Base class for all pipeline implementations."""
    
    def __init__(self):
        """Initialize base pipeline with logging."""
        super().__init__() 