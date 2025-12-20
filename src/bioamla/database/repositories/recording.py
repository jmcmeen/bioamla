# database/repositories/recording.py
"""Recording repository for database operations."""
from typing import List, Optional
from uuid import UUID

from sqlmodel import Session, select

from ..models import Recording, RecordingCreate, RecordingUpdate
from ..repository import BaseRepository


class RecordingRepository(BaseRepository[Recording, RecordingCreate, RecordingUpdate]):
    """Repository for Recording entities."""

    def __init__(self, session: Session):
        super().__init__(Recording, session)

    def get_by_project(
        self,
        project_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Recording]:
        """Get all recordings for a project."""
        statement = (
            select(Recording)
            .where(Recording.project_id == project_id)
            .offset(skip)
            .limit(limit)
        )
        return list(self.session.exec(statement).all())

    def get_by_filename(self, filename: str) -> Optional[Recording]:
        """Get a recording by its filename."""
        return self.get_by_field("filename", filename)

    def get_by_file_path(self, file_path: str) -> Optional[Recording]:
        """Get a recording by its file path."""
        return self.get_by_field("file_path", file_path)

    def get_by_hash(self, file_hash: str) -> Optional[Recording]:
        """Get a recording by its file hash (useful for deduplication)."""
        return self.get_by_field("file_hash", file_hash)

    def search_by_filename(self, pattern: str, limit: int = 50) -> List[Recording]:
        """Search recordings by filename pattern."""
        statement = (
            select(Recording)
            .where(Recording.filename.ilike(f"%{pattern}%"))
            .limit(limit)
        )
        return list(self.session.exec(statement).all())

    def get_by_location(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> List[Recording]:
        """Get recordings within a geographic bounding box."""
        statement = select(Recording).where(
            Recording.latitude >= lat_min,
            Recording.latitude <= lat_max,
            Recording.longitude >= lon_min,
            Recording.longitude <= lon_max,
        )
        return list(self.session.exec(statement).all())

    def get_with_annotations(self, recording_id: UUID) -> Optional[Recording]:
        """Get a recording with its annotations eagerly loaded."""
        from sqlalchemy.orm import selectinload

        statement = (
            select(Recording)
            .where(Recording.id == recording_id)
            .options(selectinload(Recording.annotations))
        )
        return self.session.exec(statement).first()

    def get_with_detections(self, recording_id: UUID) -> Optional[Recording]:
        """Get a recording with its detections eagerly loaded."""
        from sqlalchemy.orm import selectinload

        statement = (
            select(Recording)
            .where(Recording.id == recording_id)
            .options(selectinload(Recording.detections))
        )
        return self.session.exec(statement).first()

    def count_by_project(self, project_id: UUID) -> int:
        """Count recordings in a project."""
        from sqlalchemy import func

        statement = (
            select(func.count())
            .select_from(Recording)
            .where(Recording.project_id == project_id)
        )
        return self.session.exec(statement).one()
