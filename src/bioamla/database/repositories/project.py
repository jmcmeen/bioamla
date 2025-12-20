# database/repositories/project.py
"""Project repository for database operations."""
from typing import List, Optional
from uuid import UUID

from sqlmodel import Session, select

from ..models import Project, ProjectCreate, ProjectUpdate
from ..repository import BaseRepository


class ProjectRepository(BaseRepository[Project, ProjectCreate, ProjectUpdate]):
    """Repository for Project entities."""

    def __init__(self, session: Session):
        super().__init__(Project, session)

    def get_by_name(self, name: str) -> Optional[Project]:
        """Get a project by its name."""
        return self.get_by_field("name", name)

    def get_by_root_path(self, root_path: str) -> Optional[Project]:
        """Get a project by its root path."""
        return self.get_by_field("root_path", root_path)

    def search_by_name(self, name_pattern: str, limit: int = 10) -> List[Project]:
        """Search projects by name pattern (case-insensitive contains)."""
        statement = (
            select(Project)
            .where(Project.name.ilike(f"%{name_pattern}%"))
            .limit(limit)
        )
        return list(self.session.exec(statement).all())

    def get_with_recordings(self, project_id: UUID) -> Optional[Project]:
        """Get a project with its recordings eagerly loaded."""
        from sqlalchemy.orm import selectinload

        statement = (
            select(Project)
            .where(Project.id == project_id)
            .options(selectinload(Project.recordings))
        )
        return self.session.exec(statement).first()
