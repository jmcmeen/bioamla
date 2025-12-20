# core/run_store.py
"""
JSON-based Run Storage
======================

Stores analysis runs as JSON files in the project's .bioamla/runs directory.

Usage:
    from bioamla.core.run_store import RunStore

    store = RunStore()  # Auto-detects project

    # Create a run
    run_id = store.create_run(
        name="Frog detection",
        action="predict",
        controller="InferenceController",
        input_path="./audio/",
        parameters={"model": "birdnet", "threshold": 0.5}
    )

    # Update run
    store.update_run(run_id, status="completed", results={"detections": 42})

    # List runs
    runs = store.list_runs(action="predict", limit=10)
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class RunRecord:
    """A run record stored as JSON."""

    run_id: str
    name: str
    action: str
    status: str = "running"

    # Timing
    started_at: str = ""
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None

    # Input/Output
    input_path: str = ""
    output_path: str = ""

    # Controller and configuration
    controller: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Results
    results: Dict[str, Any] = field(default_factory=dict)
    output_files: List[str] = field(default_factory=list)

    # Error tracking
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunRecord":
        """Create from dictionary."""
        return cls(
            run_id=data.get("run_id", ""),
            name=data.get("name", ""),
            action=data.get("action", ""),
            status=data.get("status", "unknown"),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at"),
            duration_seconds=data.get("duration_seconds"),
            input_path=data.get("input_path", ""),
            output_path=data.get("output_path", ""),
            controller=data.get("controller", ""),
            parameters=data.get("parameters", {}),
            results=data.get("results", {}),
            output_files=data.get("output_files", []),
            error_message=data.get("error_message"),
        )


class RunStore:
    """
    JSON-based run storage.

    Stores runs as individual JSON files in the project's runs directory.
    Each run has its own directory: .bioamla/runs/<run_id>/run.json
    """

    def __init__(self, project_path: Optional[Path] = None):
        """
        Initialize run store.

        Args:
            project_path: Project root path (auto-detects if None)
        """
        self._project_path = project_path
        self._runs_path: Optional[Path] = None

    @property
    def runs_path(self) -> Optional[Path]:
        """Get the runs directory path."""
        if self._runs_path is None:
            from bioamla.core.project import load_project

            info = load_project(self._project_path)
            if info is not None:
                self._runs_path = info.runs_path

        return self._runs_path

    def is_available(self) -> bool:
        """Check if run storage is available (in a project)."""
        return self.runs_path is not None

    def create_run(
        self,
        name: str,
        action: str,
        controller: str = "",
        input_path: str = "",
        output_path: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Create a new run.

        Args:
            name: Run name/description
            action: Action type (predict, embed, cluster, indices, ribbit, etc.)
            controller: Controller class name
            input_path: Input file/directory
            output_path: Output file/directory
            parameters: Run parameters

        Returns:
            Run ID if successful, None if not in a project
        """
        if not self.is_available():
            return None

        # Generate run ID
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid4())[:8]

        # Create run directory
        run_dir = self.runs_path / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create run record
        run = RunRecord(
            run_id=run_id,
            name=name,
            action=action,
            status="running",
            started_at=datetime.now().isoformat(),
            controller=controller,
            input_path=input_path,
            output_path=output_path,
            parameters=parameters or {},
        )

        # Save
        self._save_run(run)
        return run_id

    def update_run(
        self,
        run_id: str,
        status: Optional[str] = None,
        results: Optional[Dict[str, Any]] = None,
        output_files: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Update a run.

        Args:
            run_id: Run ID to update
            status: New status (completed, failed, cancelled)
            results: Run results
            output_files: List of output file paths
            output_path: Output path
            error_message: Error message if failed

        Returns:
            True if successful
        """
        run = self.get_run(run_id)
        if run is None:
            return False

        if status is not None:
            run.status = status

        if results is not None:
            run.results = results

        if output_files is not None:
            run.output_files = output_files

        if output_path is not None:
            run.output_path = output_path

        if error_message is not None:
            run.error_message = error_message

        # Calculate duration if completing
        if status in ("completed", "failed", "cancelled"):
            run.completed_at = datetime.now().isoformat()
            if run.started_at:
                try:
                    started = datetime.fromisoformat(run.started_at)
                    completed = datetime.fromisoformat(run.completed_at)
                    run.duration_seconds = (completed - started).total_seconds()
                except ValueError:
                    pass

        return self._save_run(run)

    def complete_run(
        self,
        run_id: str,
        results: Optional[Dict[str, Any]] = None,
        output_files: Optional[List[str]] = None,
    ) -> bool:
        """Mark a run as completed."""
        return self.update_run(
            run_id,
            status="completed",
            results=results,
            output_files=output_files,
        )

    def fail_run(self, run_id: str, error_message: str) -> bool:
        """Mark a run as failed."""
        return self.update_run(
            run_id,
            status="failed",
            error_message=error_message,
        )

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        """
        Get a run by ID.

        Args:
            run_id: Run ID

        Returns:
            RunRecord or None if not found
        """
        if not self.is_available():
            return None

        run_file = self.runs_path / run_id / "run.json"
        if not run_file.exists():
            return None

        try:
            data = json.loads(run_file.read_text())
            return RunRecord.from_dict(data)
        except (json.JSONDecodeError, IOError):
            return None

    def list_runs(
        self,
        action: Optional[str] = None,
        status: Optional[str] = None,
        controller: Optional[str] = None,
        limit: int = 50,
    ) -> List[RunRecord]:
        """
        List runs with optional filtering.

        Args:
            action: Filter by action type
            status: Filter by status
            controller: Filter by controller
            limit: Maximum number to return

        Returns:
            List of RunRecords (newest first)
        """
        if not self.is_available():
            return []

        runs = []
        run_dirs = sorted(self.runs_path.iterdir(), reverse=True)

        for run_dir in run_dirs:
            if not run_dir.is_dir():
                continue

            run = self.get_run(run_dir.name)
            if run is None:
                continue

            # Apply filters
            if action and run.action != action:
                continue
            if status and run.status != status:
                continue
            if controller and run.controller != controller:
                continue

            runs.append(run)

            if len(runs) >= limit:
                break

        return runs

    def save_artifact(
        self,
        run_id: str,
        filename: str,
        data: Any,
    ) -> Optional[Path]:
        """
        Save an artifact to the run directory.

        Args:
            run_id: Run ID
            filename: Artifact filename
            data: Data to save (dict/list for JSON, str for text)

        Returns:
            Path to saved file or None
        """
        if not self.is_available():
            return None

        run_dir = self.runs_path / run_id
        if not run_dir.exists():
            return None

        file_path = run_dir / filename

        try:
            if isinstance(data, (dict, list)):
                file_path.write_text(json.dumps(data, indent=2, default=str))
            elif isinstance(data, str):
                file_path.write_text(data)
            else:
                file_path.write_text(json.dumps(data, indent=2, default=str))

            return file_path
        except IOError:
            return None

    def get_artifact(self, run_id: str, filename: str) -> Optional[Any]:
        """
        Load an artifact from the run directory.

        Args:
            run_id: Run ID
            filename: Artifact filename

        Returns:
            Loaded data or None
        """
        if not self.is_available():
            return None

        file_path = self.runs_path / run_id / filename
        if not file_path.exists():
            return None

        try:
            content = file_path.read_text()
            if filename.endswith(".json"):
                return json.loads(content)
            return content
        except (IOError, json.JSONDecodeError):
            return None

    def delete_run(self, run_id: str) -> bool:
        """
        Delete a run and all its artifacts.

        Args:
            run_id: Run ID to delete

        Returns:
            True if successful
        """
        import shutil

        if not self.is_available():
            return False

        run_dir = self.runs_path / run_id
        if not run_dir.exists():
            return False

        try:
            shutil.rmtree(run_dir)
            return True
        except IOError:
            return False

    def _save_run(self, run: RunRecord) -> bool:
        """Save a run record to disk."""
        if not self.is_available():
            return False

        run_dir = self.runs_path / run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        run_file = run_dir / "run.json"
        try:
            run_file.write_text(json.dumps(run.to_dict(), indent=2))
            return True
        except IOError:
            return False


# Convenience function for quick access
def get_run_store(project_path: Optional[Path] = None) -> RunStore:
    """Get a run store instance."""
    return RunStore(project_path)
