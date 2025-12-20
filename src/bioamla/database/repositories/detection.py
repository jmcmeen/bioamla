# database/repositories/detection.py
"""Detection repository for database operations."""
from typing import List, Optional
from uuid import UUID

from sqlmodel import Session, select

from ..models import Detection, DetectionCreate, DetectionUpdate
from ..repository import BaseRepository


class DetectionRepository(BaseRepository[Detection, DetectionCreate, DetectionUpdate]):
    """Repository for Detection entities."""

    def __init__(self, session: Session):
        super().__init__(Detection, session)

    def get_by_recording(
        self,
        recording_id: UUID,
        skip: int = 0,
        limit: int = 1000,
    ) -> List[Detection]:
        """Get all detections for a recording."""
        statement = (
            select(Detection)
            .where(Detection.recording_id == recording_id)
            .order_by(Detection.start_time)
            .offset(skip)
            .limit(limit)
        )
        return list(self.session.exec(statement).all())

    def get_by_label(
        self,
        predicted_label: str,
        min_confidence: float = 0.0,
        limit: int = 100,
    ) -> List[Detection]:
        """Get detections with a specific predicted label."""
        statement = (
            select(Detection)
            .where(
                Detection.predicted_label == predicted_label,
                Detection.confidence >= min_confidence,
            )
            .order_by(Detection.confidence.desc())
            .limit(limit)
        )
        return list(self.session.exec(statement).all())

    def get_by_model(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        limit: int = 100,
    ) -> List[Detection]:
        """Get detections from a specific model."""
        statement = select(Detection).where(Detection.model_name == model_name)
        if model_version:
            statement = statement.where(Detection.model_version == model_version)
        statement = statement.limit(limit)
        return list(self.session.exec(statement).all())

    def get_high_confidence(
        self,
        recording_id: UUID,
        min_confidence: float = 0.9,
    ) -> List[Detection]:
        """Get high-confidence detections for a recording."""
        statement = (
            select(Detection)
            .where(
                Detection.recording_id == recording_id,
                Detection.confidence >= min_confidence,
            )
            .order_by(Detection.confidence.desc())
        )
        return list(self.session.exec(statement).all())

    def get_unverified(
        self,
        recording_id: Optional[UUID] = None,
        limit: int = 100,
    ) -> List[Detection]:
        """Get unverified detections."""
        statement = select(Detection).where(Detection.verified == False)
        if recording_id:
            statement = statement.where(Detection.recording_id == recording_id)
        statement = statement.order_by(Detection.confidence.desc()).limit(limit)
        return list(self.session.exec(statement).all())

    def get_verified(
        self,
        recording_id: Optional[UUID] = None,
        limit: int = 100,
    ) -> List[Detection]:
        """Get verified detections."""
        statement = select(Detection).where(Detection.verified == True)
        if recording_id:
            statement = statement.where(Detection.recording_id == recording_id)
        statement = statement.limit(limit)
        return list(self.session.exec(statement).all())

    def verify(self, detection_id: UUID, verified_label: str) -> Optional[Detection]:
        """Mark a detection as verified with a corrected label."""
        return self.update_by_id(
            detection_id,
            {"verified": True, "verified_label": verified_label},
        )

    def get_labels(self, min_confidence: float = 0.0) -> List[str]:
        """Get distinct predicted labels."""
        from sqlalchemy import distinct

        statement = (
            select(distinct(Detection.predicted_label))
            .where(Detection.confidence >= min_confidence)
        )
        return list(self.session.exec(statement).all())

    def count_by_recording(self, recording_id: UUID) -> int:
        """Count detections for a recording."""
        from sqlalchemy import func

        statement = (
            select(func.count())
            .select_from(Detection)
            .where(Detection.recording_id == recording_id)
        )
        return self.session.exec(statement).one()

    def count_by_label(self, predicted_label: str) -> int:
        """Count detections with a specific label."""
        from sqlalchemy import func

        statement = (
            select(func.count())
            .select_from(Detection)
            .where(Detection.predicted_label == predicted_label)
        )
        return self.session.exec(statement).one()
