# database/unit_of_work.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from sqlmodel import Session
from contextlib import contextmanager

from .connection import DatabaseConnection

if TYPE_CHECKING:
    from typing import Generator


class UnitOfWork:
    """
    Unit of Work pattern implementation.

    Manages database transactions and provides access to repositories.
    Ensures all operations within a unit are committed or rolled back together.

    Usage:
        with UnitOfWork(db) as uow:
            item = uow.items.create(item_data)
            uow.commit()

    Or as a context manager that auto-commits:
        with UnitOfWork(db).auto_commit() as uow:
            item = uow.items.create(item_data)
            # Automatically committed on exit
    """

    def __init__(self, database: DatabaseConnection):
        self._database = database
        self._session: Session | None = None

        # Repository instances (lazily initialized)
        # Add your repository properties here, e.g.:
        # self._users: UserRepository | None = None

    @property
    def session(self) -> Session:
        """Get the current session."""
        if self._session is None:
            raise RuntimeError("UnitOfWork not started. Use 'with' statement.")
        return self._session

    # Add repository properties here, e.g.:
    # @property
    # def users(self) -> UserRepository:
    #     """User repository instance."""
    #     if self._users is None:
    #         self._users = UserRepository(self.session)
    #     return self._users

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
            # Reset repository instances here

    def refresh(self, obj) -> None:
        """Refresh an object from the database."""
        self.session.refresh(obj)

    @contextmanager
    def auto_commit(self) -> Generator[UnitOfWork, None, None]:
        """
        Context manager that auto-commits on successful exit.

        Usage:
            with UnitOfWork(db).auto_commit() as uow:
                uow.items.create(...)
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

    # Declare repository attributes here, e.g.:
    # users: UserRepository
    # projects: ProjectRepository

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
