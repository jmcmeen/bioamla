# database/models.py
"""
SQLModel domain models for bioamla database layer.

Models:
    - Project: Represents a bioamla project with metadata
    - Recording: Audio file record with metadata
    - Annotation: Time-frequency annotation (detection or manual label)
    - Detection: Model prediction result
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlmodel import JSON, Column, Field, Relationship, SQLModel


class TimestampMixin(SQLModel):
    """Mixin for created/updated timestamps."""

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default=None)


# =============================================================================
# Project Model
# =============================================================================


class ProjectBase(SQLModel):
    """Shared project properties."""

    name: str = Field(index=True, max_length=255)
    description: Optional[str] = Field(default=None, max_length=1000)
    version: str = Field(default="1.0.0", max_length=50)
    root_path: str = Field(max_length=1024)  # Absolute path to project root


class Project(ProjectBase, TimestampMixin, table=True):
    """Project database model."""

    __tablename__ = "projects"

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)

    # Relationships
    recordings: List["Recording"] = Relationship(back_populates="project")


class ProjectCreate(ProjectBase):
    """Schema for creating a project."""

    pass


class ProjectUpdate(SQLModel):
    """Schema for updating a project (all fields optional)."""

    name: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None


# =============================================================================
# Recording Model
# =============================================================================


class RecordingBase(SQLModel):
    """Shared recording properties."""

    filename: str = Field(index=True, max_length=512)
    file_path: str = Field(max_length=2048)  # Relative path from project root
    duration_seconds: Optional[float] = Field(default=None)
    sample_rate: Optional[int] = Field(default=None)
    channels: Optional[int] = Field(default=None)
    bit_depth: Optional[int] = Field(default=None)
    file_size_bytes: Optional[int] = Field(default=None)
    file_hash: Optional[str] = Field(default=None, max_length=64)  # SHA-256

    # Metadata
    recording_date: Optional[datetime] = Field(default=None)
    location_name: Optional[str] = Field(default=None, max_length=255)
    latitude: Optional[float] = Field(default=None)
    longitude: Optional[float] = Field(default=None)
    notes: Optional[str] = Field(default=None)


class Recording(RecordingBase, TimestampMixin, table=True):
    """Recording database model."""

    __tablename__ = "recordings"

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    project_id: Optional[UUID] = Field(default=None, foreign_key="projects.id", index=True)

    # Relationships
    project: Optional[Project] = Relationship(back_populates="recordings")
    annotations: List["Annotation"] = Relationship(back_populates="recording")
    detections: List["Detection"] = Relationship(back_populates="recording")


class RecordingCreate(RecordingBase):
    """Schema for creating a recording."""

    project_id: Optional[UUID] = None


class RecordingUpdate(SQLModel):
    """Schema for updating a recording (all fields optional)."""

    filename: Optional[str] = None
    file_path: Optional[str] = None
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    recording_date: Optional[datetime] = None
    location_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    notes: Optional[str] = None


# =============================================================================
# Annotation Model
# =============================================================================


class AnnotationBase(SQLModel):
    """
    Shared annotation properties.

    Mirrors the existing Annotation dataclass in bioamla.core.annotations
    for database persistence.
    """

    start_time: float = Field(index=True)  # Start time in seconds
    end_time: float  # End time in seconds
    low_freq: Optional[float] = Field(default=None)  # Lower frequency bound (Hz)
    high_freq: Optional[float] = Field(default=None)  # Upper frequency bound (Hz)
    label: str = Field(default="", index=True, max_length=255)
    channel: int = Field(default=1)
    confidence: Optional[float] = Field(default=None)
    notes: Optional[str] = Field(default=None)
    custom_fields: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    # Source tracking
    source: Optional[str] = Field(default=None, max_length=50)  # 'manual', 'model', 'import'
    source_file: Optional[str] = Field(default=None, max_length=512)  # Original annotation file


class Annotation(AnnotationBase, TimestampMixin, table=True):
    """Annotation database model."""

    __tablename__ = "annotations"

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    recording_id: Optional[UUID] = Field(default=None, foreign_key="recordings.id", index=True)

    # Relationships
    recording: Optional[Recording] = Relationship(back_populates="annotations")


class AnnotationCreate(AnnotationBase):
    """Schema for creating an annotation."""

    recording_id: Optional[UUID] = None


class AnnotationUpdate(SQLModel):
    """Schema for updating an annotation (all fields optional)."""

    start_time: Optional[float] = None
    end_time: Optional[float] = None
    low_freq: Optional[float] = None
    high_freq: Optional[float] = None
    label: Optional[str] = None
    channel: Optional[int] = None
    confidence: Optional[float] = None
    notes: Optional[str] = None
    custom_fields: Optional[Dict[str, Any]] = None


# =============================================================================
# Detection Model
# =============================================================================


class DetectionBase(SQLModel):
    """
    Shared detection properties.

    Represents a model prediction result, linked to a recording and
    optionally to a specific annotation region.
    """

    start_time: float = Field(index=True)
    end_time: float
    low_freq: Optional[float] = Field(default=None)
    high_freq: Optional[float] = Field(default=None)

    # Prediction results
    predicted_label: str = Field(index=True, max_length=255)
    confidence: float = Field(default=0.0)  # Primary confidence score
    top_k_labels: Optional[List[str]] = Field(default=None, sa_column=Column(JSON))
    top_k_scores: Optional[List[float]] = Field(default=None, sa_column=Column(JSON))

    # Model info
    model_name: Optional[str] = Field(default=None, max_length=255)
    model_version: Optional[str] = Field(default=None, max_length=50)

    # Verification status
    verified: bool = Field(default=False)
    verified_label: Optional[str] = Field(default=None, max_length=255)


class Detection(DetectionBase, TimestampMixin, table=True):
    """Detection database model."""

    __tablename__ = "detections"

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    recording_id: Optional[UUID] = Field(default=None, foreign_key="recordings.id", index=True)

    # Relationships
    recording: Optional[Recording] = Relationship(back_populates="detections")


class DetectionCreate(DetectionBase):
    """Schema for creating a detection."""

    recording_id: Optional[UUID] = None


class DetectionUpdate(SQLModel):
    """Schema for updating a detection (all fields optional)."""

    predicted_label: Optional[str] = None
    confidence: Optional[float] = None
    verified: Optional[bool] = None
    verified_label: Optional[str] = None
