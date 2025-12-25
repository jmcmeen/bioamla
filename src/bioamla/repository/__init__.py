"""File repository layer for dependency injection."""

from bioamla.repository.local import LocalFileRepository
from bioamla.repository.protocol import FileRepositoryProtocol

__all__ = [
    "FileRepositoryProtocol",
    "LocalFileRepository",
]
