# src/core/m_detector/__init__.py
from .base import MDetector
from ..constants import OcclusionResult

# Export the main class and enums at the module level
__all__ = ['MDetector', 'OcclusionResult']