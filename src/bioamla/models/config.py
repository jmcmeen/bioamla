# models/config.py
"""
Data models for configuration operations.
"""

from dataclasses import dataclass
from typing import List, Optional

from .base import ToDictMixin


@dataclass
class ConfigInfo(ToDictMixin):
    """Information about the current configuration."""

    source: Optional[str]
    is_default: bool
    sections: List[str]
