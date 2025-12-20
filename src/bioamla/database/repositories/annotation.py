# database/repositories/annotation.py
"""Annotation repository for database operations."""
from typing import List, Optional
from uuid import UUID

from sqlmodel import Session, select

from ..models import Annotation, AnnotationCreate, AnnotationUpdate
from ..repository import BaseRepository


class AnnotationRepository(BaseRepository[Annotation, AnnotationCreate, AnnotationUpdate]):
    """Repository for Annotation entities."""

    def __init__(self, session: Session):
        super().__init__(Annotation, session)

    def get_by_recording(
        self,
        recording_id: UUID,
        skip: int = 0,
        limit: int = 1000,
    ) -> List[Annotation]:
        """Get all annotations for a recording."""
        statement = (
            select(Annotation)
            .where(Annotation.recording_id == recording_id)
            .order_by(Annotation.start_time)
            .offset(skip)
            .limit(limit)
        )
        return list(self.session.exec(statement).all())

    def get_by_label(self, label: str, limit: int = 100) -> List[Annotation]:
        """Get annotations with a specific label."""
        statement = (
            select(Annotation)
            .where(Annotation.label == label)
            .limit(limit)
        )
        return list(self.session.exec(statement).all())

    def get_by_time_range(
        self,
        recording_id: UUID,
        start_time: float,
        end_time: float,
    ) -> List[Annotation]:
        """Get annotations within a time range for a recording."""
        statement = (
            select(Annotation)
            .where(
                Annotation.recording_id == recording_id,
                Annotation.start_time >= start_time,
                Annotation.end_time <= end_time,
            )
            .order_by(Annotation.start_time)
        )
        return list(self.session.exec(statement).all())

    def get_overlapping(
        self,
        recording_id: UUID,
        start_time: float,
        end_time: float,
    ) -> List[Annotation]:
        """Get annotations that overlap with a time range."""
        statement = (
            select(Annotation)
            .where(
                Annotation.recording_id == recording_id,
                Annotation.start_time < end_time,
                Annotation.end_time > start_time,
            )
            .order_by(Annotation.start_time)
        )
        return list(self.session.exec(statement).all())

    def get_labels(self, recording_id: Optional[UUID] = None) -> List[str]:
        """Get distinct labels, optionally filtered by recording."""
        from sqlalchemy import distinct

        statement = select(distinct(Annotation.label))
        if recording_id:
            statement = statement.where(Annotation.recording_id == recording_id)
        return list(self.session.exec(statement).all())

    def count_by_label(self, label: str) -> int:
        """Count annotations with a specific label."""
        from sqlalchemy import func

        statement = (
            select(func.count())
            .select_from(Annotation)
            .where(Annotation.label == label)
        )
        return self.session.exec(statement).one()

    def count_by_recording(self, recording_id: UUID) -> int:
        """Count annotations for a recording."""
        from sqlalchemy import func

        statement = (
            select(func.count())
            .select_from(Annotation)
            .where(Annotation.recording_id == recording_id)
        )
        return self.session.exec(statement).one()

    def get_by_source(self, source: str, limit: int = 100) -> List[Annotation]:
        """Get annotations by source (manual, model, import)."""
        statement = (
            select(Annotation)
            .where(Annotation.source == source)
            .limit(limit)
        )
        return list(self.session.exec(statement).all())
