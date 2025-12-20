# database/__init__.py
"""
Bioamla Database Layer
======================

SQLModel-based database layer with Unit of Work pattern.

Components:
    - DatabaseConnection: Engine and session management
    - BaseRepository: Generic CRUD operations
    - UnitOfWork: Transaction management with repository access
    - Models: Project, Recording, Annotation, Detection

Usage:
    from bioamla.database import get_database, UnitOfWork
    from bioamla.database.models import ProjectCreate

    db = get_database("sqlite:///bioamla.db")
    db.create_tables()

    with UnitOfWork(db).auto_commit() as uow:
        project = uow.projects.create(ProjectCreate(
            name="My Project",
            root_path="/path/to/project"
        ))
        print(f"Created project: {project.id}")
"""

from .connection import DatabaseConnection, get_database
from .models import (
    Annotation,
    AnnotationCreate,
    AnnotationUpdate,
    Detection,
    DetectionCreate,
    DetectionUpdate,
    Project,
    ProjectCreate,
    ProjectUpdate,
    Recording,
    RecordingCreate,
    RecordingUpdate,
    TimestampMixin,
)
from .repositories import (
    AnnotationRepository,
    DetectionRepository,
    ProjectRepository,
    RecordingRepository,
)
from .repository import BaseRepository
from .unit_of_work import AbstractUnitOfWork, UnitOfWork

__all__ = [
    # Connection
    "DatabaseConnection",
    "get_database",
    # Base
    "BaseRepository",
    "UnitOfWork",
    "AbstractUnitOfWork",
    "TimestampMixin",
    # Models
    "Project",
    "ProjectCreate",
    "ProjectUpdate",
    "Recording",
    "RecordingCreate",
    "RecordingUpdate",
    "Annotation",
    "AnnotationCreate",
    "AnnotationUpdate",
    "Detection",
    "DetectionCreate",
    "DetectionUpdate",
    # Repositories
    "ProjectRepository",
    "RecordingRepository",
    "AnnotationRepository",
    "DetectionRepository",
]
