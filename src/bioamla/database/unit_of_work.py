# database/unit_of_work.py
"""Unit of Work pattern implementation for bioamla database layer."""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING

from sqlmodel import Session

from .connection import DatabaseConnection
from .repositories import (
    AnnotationRepository,
    DetectionRepository,
    ProjectRepository,
    RecordingRepository,
)

if TYPE_CHECKING:
    from typing import Generator


class UnitOfWork:
    """
    Unit of Work pattern implementation.

    Manages database transactions and provides access to repositories.
    Ensures all operations within a unit are committed or rolled back together.

    Usage:
        with UnitOfWork(db) as uow:
            project = uow.projects.create(project_data)
            uow.commit()

    Or as a context manager that auto-commits:
        with UnitOfWork(db).auto_commit() as uow:
            project = uow.projects.create(project_data)
            # Automatically committed on exit
    """

    def __init__(self, database: DatabaseConnection):
        self._database = database
        self._session: Session | None = None

        # Repository instances (lazily initialized)
        self._projects: ProjectRepository | None = None
        self._recordings: RecordingRepository | None = None
        self._annotations: AnnotationRepository | None = None
        self._detections: DetectionRepository | None = None

    @property
    def session(self) -> Session:
        """Get the current session."""
        if self._session is None:
            raise RuntimeError("UnitOfWork not started. Use 'with' statement.")
        return self._session

    @property
    def projects(self) -> ProjectRepository:
        """Project repository instance."""
        if self._projects is None:
            self._projects = ProjectRepository(self.session)
        return self._projects

    @property
    def recordings(self) -> RecordingRepository:
        """Recording repository instance."""
        if self._recordings is None:
            self._recordings = RecordingRepository(self.session)
        return self._recordings

    @property
    def annotations(self) -> AnnotationRepository:
        """Annotation repository instance."""
        if self._annotations is None:
            self._annotations = AnnotationRepository(self.session)
        return self._annotations

    @property
    def detections(self) -> DetectionRepository:
        """Detection repository instance."""
        if self._detections is None:
            self._detections = DetectionRepository(self.session)
        return self._detections

    def __enter__(self) -> UnitOfWork:
        """Start the unit of work."""
        self._session = Session(self._database.engine)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End the unit of work, rolling back if there was an exception."""
        if exc_type is not None:
            self.rollback()
        self.close()

    def commit(self) -> None:
        """Commit the current transaction."""
        self.session.commit()

    def rollback(self) -> None:
        """Rollback the current transaction."""
        self.session.rollback()

    def close(self) -> None:
        """Close the session."""
        if self._session is not None:
            self._session.close()
            self._session = None
            # Reset repository instances
            self._projects = None
            self._recordings = None
            self._annotations = None
            self._detections = None

    def refresh(self, obj) -> None:
        """Refresh an object from the database."""
        self.session.refresh(obj)

    @contextmanager
    def auto_commit(self) -> Generator[UnitOfWork, None, None]:
        """
        Context manager that auto-commits on successful exit.

        Usage:
            with UnitOfWork(db).auto_commit() as uow:
                uow.projects.create(...)
                # Auto-committed here
        """
        self._session = Session(self._database.engine)
        try:
            yield self
            self.commit()
        except Exception:
            self.rollback()
            raise
        finally:
            self.close()


class AbstractUnitOfWork(ABC):
    """Abstract UoW for dependency injection and testing."""

    projects: ProjectRepository
    recordings: RecordingRepository
    annotations: AnnotationRepository
    detections: DetectionRepository

    @abstractmethod
    def commit(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def rollback(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def __enter__(self) -> "AbstractUnitOfWork":
        raise NotImplementedError

    @abstractmethod
    def __exit__(self, *args) -> None:
        raise NotImplementedError
