"""Dependency tracking and reporting models."""

from dataclasses import dataclass
from typing import List, Optional

from bioamla.models.base import ToDictMixin


@dataclass
class DependencyInfo(ToDictMixin):
    """Information about a system dependency."""

    name: str
    description: str
    required_for: str
    installed: bool
    version: Optional[str] = None
    install_hint: Optional[str] = None


@dataclass
class DependencyReport(ToDictMixin):
    """Report of all dependency statuses."""

    os_type: str
    all_installed: bool
    dependencies: List[DependencyInfo]
    install_command: Optional[str] = None
