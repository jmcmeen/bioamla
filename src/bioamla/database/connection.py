# database/connection.py
from sqlmodel import SQLModel, Session, create_engine
from typing import Generator
from contextlib import contextmanager


class DatabaseConnection:
    """Manages database engine and session creation."""

    def __init__(self, url: str, echo: bool = False):
        self.engine = create_engine(url, echo=echo)

    def create_tables(self) -> None:
        """Create all tables defined in SQLModel metadata."""
        SQLModel.metadata.create_all(self.engine)

    def drop_tables(self) -> None:
        """Drop all tables (use with caution)."""
        SQLModel.metadata.drop_all(self.engine)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around operations."""
        session = Session(self.engine)
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


def get_database(url: str = "sqlite:///database.db") -> DatabaseConnection:
    """Default connection factory."""
    return DatabaseConnection(url)
