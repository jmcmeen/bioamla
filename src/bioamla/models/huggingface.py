# models/huggingface.py
"""
Data models for HuggingFace operations.
"""

from dataclasses import dataclass

from .base import ToDictMixin


@dataclass
class PushResult(ToDictMixin):
    """Result of a push operation."""

    repo_id: str
    repo_type: str
    url: str
    files_uploaded: int
    total_size_bytes: int
