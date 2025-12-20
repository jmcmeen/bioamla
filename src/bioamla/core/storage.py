# core/storage.py
"""
Multimodal Storage Interface
============================

Provides a unified interface for storing runs and artifacts with multiple backends:
- JSON: File-based storage in .bioamla/runs/ (default, no dependencies)
- SQLite: Local database storage (requires sqlite3, included in Python)
- PostgreSQL: Remote database storage (requires psycopg2)

Usage:
    from bioamla.core.storage import get_storage, StorageBackend

    # Auto-detect from project config or environment
    storage = get_storage()

    # Explicitly specify backend
    storage = get_storage(backend=StorageBackend.SQLITE)

    # Create a run
    run_id = storage.create_run(
        name="Species detection",
        action="predict",
        controller="InferenceController",
        input_path="./audio/",
        parameters={"model": "birdnet", "threshold": 0.5}
    )

    # Update and complete
    storage.update_run(run_id, status="completed", results={"detections": 42})
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


class StorageBackend(Enum):
    """Available storage backends."""

    JSON = "json"
    SQLITE = "sqlite"
    POSTGRES = "postgres"


@dataclass
class RunRecord:
    """A run record that can be stored in any backend."""

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


class BaseStorage(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if storage is available."""
        pass

    @abstractmethod
    def create_run(
        self,
        name: str,
        action: str,
        controller: str = "",
        input_path: str = "",
        output_path: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Create a new run."""
        pass

    @abstractmethod
    def update_run(
        self,
        run_id: str,
        status: Optional[str] = None,
        results: Optional[Dict[str, Any]] = None,
        output_files: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """Update a run."""
        pass

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

    @abstractmethod
    def get_run(self, run_id: str) -> Optional[RunRecord]:
        """Get a run by ID."""
        pass

    @abstractmethod
    def list_runs(
        self,
        action: Optional[str] = None,
        status: Optional[str] = None,
        controller: Optional[str] = None,
        limit: int = 50,
    ) -> List[RunRecord]:
        """List runs with optional filtering."""
        pass

    @abstractmethod
    def delete_run(self, run_id: str) -> bool:
        """Delete a run."""
        pass

    @abstractmethod
    def save_artifact(
        self,
        run_id: str,
        filename: str,
        data: Any,
    ) -> Optional[Path]:
        """Save an artifact."""
        pass

    @abstractmethod
    def get_artifact(self, run_id: str, filename: str) -> Optional[Any]:
        """Load an artifact."""
        pass

    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid4())[:8]

    def _calculate_duration(self, started_at: str, completed_at: str) -> Optional[float]:
        """Calculate duration between timestamps."""
        try:
            started = datetime.fromisoformat(started_at)
            completed = datetime.fromisoformat(completed_at)
            return (completed - started).total_seconds()
        except ValueError:
            return None


class JSONStorage(BaseStorage):
    """
    JSON-based file storage.

    Stores runs as individual JSON files in the project's runs directory.
    Each run has its own directory: .bioamla/runs/<run_id>/run.json
    """

    def __init__(self, project_path: Optional[Path] = None):
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
        """Check if JSON storage is available (in a project)."""
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
        if not self.is_available():
            return None

        run_id = self._generate_run_id()

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
                run.duration_seconds = self._calculate_duration(run.started_at, run.completed_at)

        return self._save_run(run)

    def get_run(self, run_id: str) -> Optional[RunRecord]:
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

    def delete_run(self, run_id: str) -> bool:
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

    def save_artifact(
        self,
        run_id: str,
        filename: str,
        data: Any,
    ) -> Optional[Path]:
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


class SQLiteStorage(BaseStorage):
    """
    SQLite-based database storage.

    Stores runs in a local SQLite database.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path
        self._connection = None
        self._initialized = False

    @property
    def db_path(self) -> Optional[Path]:
        """Get the database path."""
        if self._db_path is None:
            from bioamla.core.project import load_project

            info = load_project()
            if info is not None:
                self._db_path = info.cache_path / "runs.db"

        return self._db_path

    def _get_connection(self):
        """Get or create database connection."""
        import sqlite3

        if self._connection is None and self.db_path is not None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(str(self.db_path))
            self._connection.row_factory = sqlite3.Row

            if not self._initialized:
                self._init_db()
                self._initialized = True

        return self._connection

    def _init_db(self):
        """Initialize database schema."""
        conn = self._connection
        if conn is None:
            return

        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT DEFAULT 'running',
                started_at TEXT,
                completed_at TEXT,
                duration_seconds REAL,
                input_path TEXT,
                output_path TEXT,
                controller TEXT,
                parameters TEXT,
                results TEXT,
                output_files TEXT,
                error_message TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                content TEXT,
                created_at TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
                UNIQUE(run_id, filename)
            )
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_action ON runs(action)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at)")
        conn.commit()

    def is_available(self) -> bool:
        """Check if SQLite storage is available."""
        return self.db_path is not None

    def create_run(
        self,
        name: str,
        action: str,
        controller: str = "",
        input_path: str = "",
        output_path: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        conn = self._get_connection()
        if conn is None:
            return None

        run_id = self._generate_run_id()

        conn.execute(
            """
            INSERT INTO runs (run_id, name, action, status, started_at, controller,
                            input_path, output_path, parameters)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                name,
                action,
                "running",
                datetime.now().isoformat(),
                controller,
                input_path,
                output_path,
                json.dumps(parameters or {}),
            ),
        )
        conn.commit()
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
        conn = self._get_connection()
        if conn is None:
            return False

        updates = []
        values = []

        if status is not None:
            updates.append("status = ?")
            values.append(status)

            if status in ("completed", "failed", "cancelled"):
                completed_at = datetime.now().isoformat()
                updates.append("completed_at = ?")
                values.append(completed_at)

                # Calculate duration
                cursor = conn.execute("SELECT started_at FROM runs WHERE run_id = ?", (run_id,))
                row = cursor.fetchone()
                if row and row["started_at"]:
                    duration = self._calculate_duration(row["started_at"], completed_at)
                    if duration is not None:
                        updates.append("duration_seconds = ?")
                        values.append(duration)

        if results is not None:
            updates.append("results = ?")
            values.append(json.dumps(results))

        if output_files is not None:
            updates.append("output_files = ?")
            values.append(json.dumps(output_files))

        if output_path is not None:
            updates.append("output_path = ?")
            values.append(output_path)

        if error_message is not None:
            updates.append("error_message = ?")
            values.append(error_message)

        if not updates:
            return True

        values.append(run_id)
        query = f"UPDATE runs SET {', '.join(updates)} WHERE run_id = ?"

        try:
            conn.execute(query, values)
            conn.commit()
            return True
        except Exception:
            return False

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        conn = self._get_connection()
        if conn is None:
            return None

        cursor = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_record(row)

    def list_runs(
        self,
        action: Optional[str] = None,
        status: Optional[str] = None,
        controller: Optional[str] = None,
        limit: int = 50,
    ) -> List[RunRecord]:
        conn = self._get_connection()
        if conn is None:
            return []

        conditions = []
        values = []

        if action:
            conditions.append("action = ?")
            values.append(action)
        if status:
            conditions.append("status = ?")
            values.append(status)
        if controller:
            conditions.append("controller = ?")
            values.append(controller)

        query = "SELECT * FROM runs"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY started_at DESC LIMIT ?"
        values.append(limit)

        cursor = conn.execute(query, values)
        return [self._row_to_record(row) for row in cursor.fetchall()]

    def delete_run(self, run_id: str) -> bool:
        conn = self._get_connection()
        if conn is None:
            return False

        try:
            conn.execute("DELETE FROM artifacts WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            conn.commit()
            return True
        except Exception:
            return False

    def save_artifact(
        self,
        run_id: str,
        filename: str,
        data: Any,
    ) -> Optional[Path]:
        conn = self._get_connection()
        if conn is None:
            return None

        content = data if isinstance(data, str) else json.dumps(data, default=str)

        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO artifacts (run_id, filename, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, filename, content, datetime.now().isoformat()),
            )
            conn.commit()
            # Return a virtual path for consistency
            return Path(f"sqlite://{self.db_path}/artifacts/{run_id}/{filename}")
        except Exception:
            return None

    def get_artifact(self, run_id: str, filename: str) -> Optional[Any]:
        conn = self._get_connection()
        if conn is None:
            return None

        cursor = conn.execute(
            "SELECT content FROM artifacts WHERE run_id = ? AND filename = ?",
            (run_id, filename),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        content = row["content"]
        if filename.endswith(".json"):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass
        return content

    def _row_to_record(self, row) -> RunRecord:
        """Convert a database row to RunRecord."""

        def parse_json(val, default):
            if val is None:
                return default
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return default

        return RunRecord(
            run_id=row["run_id"],
            name=row["name"],
            action=row["action"],
            status=row["status"] or "unknown",
            started_at=row["started_at"] or "",
            completed_at=row["completed_at"],
            duration_seconds=row["duration_seconds"],
            input_path=row["input_path"] or "",
            output_path=row["output_path"] or "",
            controller=row["controller"] or "",
            parameters=parse_json(row["parameters"], {}),
            results=parse_json(row["results"], {}),
            output_files=parse_json(row["output_files"], []),
            error_message=row["error_message"],
        )


class PostgresStorage(BaseStorage):
    """
    PostgreSQL-based database storage.

    Stores runs in a remote PostgreSQL database.
    Requires psycopg2 or psycopg2-binary.
    """

    def __init__(self, connection_string: Optional[str] = None):
        self._connection_string = connection_string or os.environ.get("BIOAMLA_DB_URL")
        self._connection = None
        self._initialized = False

    def _get_connection(self):
        """Get or create database connection."""
        if self._connection is None and self._connection_string:
            try:
                import psycopg2
                from psycopg2.extras import RealDictCursor

                self._connection = psycopg2.connect(
                    self._connection_string, cursor_factory=RealDictCursor
                )

                if not self._initialized:
                    self._init_db()
                    self._initialized = True

            except ImportError:
                raise ImportError(
                    "PostgreSQL support requires psycopg2. "
                    "Install with: pip install psycopg2-binary"
                )

        return self._connection

    def _init_db(self):
        """Initialize database schema."""
        conn = self._connection
        if conn is None:
            return

        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    action TEXT NOT NULL,
                    status TEXT DEFAULT 'running',
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    duration_seconds DOUBLE PRECISION,
                    input_path TEXT,
                    output_path TEXT,
                    controller TEXT,
                    parameters JSONB,
                    results JSONB,
                    output_files JSONB,
                    error_message TEXT
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    id SERIAL PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    content TEXT,
                    created_at TIMESTAMPTZ,
                    UNIQUE(run_id, filename)
                )
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_action ON runs(action)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at)")

        conn.commit()

    def is_available(self) -> bool:
        """Check if PostgreSQL storage is available."""
        if not self._connection_string:
            return False
        try:
            self._get_connection()
            return self._connection is not None
        except Exception:
            return False

    def create_run(
        self,
        name: str,
        action: str,
        controller: str = "",
        input_path: str = "",
        output_path: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        conn = self._get_connection()
        if conn is None:
            return None

        run_id = self._generate_run_id()

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (run_id, name, action, status, started_at, controller,
                                input_path, output_path, parameters)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    name,
                    action,
                    "running",
                    datetime.now().isoformat(),
                    controller,
                    input_path,
                    output_path,
                    json.dumps(parameters or {}),
                ),
            )
        conn.commit()
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
        conn = self._get_connection()
        if conn is None:
            return False

        updates = []
        values = []

        if status is not None:
            updates.append("status = %s")
            values.append(status)

            if status in ("completed", "failed", "cancelled"):
                completed_at = datetime.now().isoformat()
                updates.append("completed_at = %s")
                values.append(completed_at)

                # Calculate duration
                with conn.cursor() as cur:
                    cur.execute("SELECT started_at FROM runs WHERE run_id = %s", (run_id,))
                    row = cur.fetchone()
                    if row and row["started_at"]:
                        started_str = row["started_at"]
                        if hasattr(started_str, "isoformat"):
                            started_str = started_str.isoformat()
                        duration = self._calculate_duration(started_str, completed_at)
                        if duration is not None:
                            updates.append("duration_seconds = %s")
                            values.append(duration)

        if results is not None:
            updates.append("results = %s")
            values.append(json.dumps(results))

        if output_files is not None:
            updates.append("output_files = %s")
            values.append(json.dumps(output_files))

        if output_path is not None:
            updates.append("output_path = %s")
            values.append(output_path)

        if error_message is not None:
            updates.append("error_message = %s")
            values.append(error_message)

        if not updates:
            return True

        values.append(run_id)
        query = f"UPDATE runs SET {', '.join(updates)} WHERE run_id = %s"

        try:
            with conn.cursor() as cur:
                cur.execute(query, values)
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            return False

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        conn = self._get_connection()
        if conn is None:
            return None

        with conn.cursor() as cur:
            cur.execute("SELECT * FROM runs WHERE run_id = %s", (run_id,))
            row = cur.fetchone()

        if row is None:
            return None

        return self._row_to_record(row)

    def list_runs(
        self,
        action: Optional[str] = None,
        status: Optional[str] = None,
        controller: Optional[str] = None,
        limit: int = 50,
    ) -> List[RunRecord]:
        conn = self._get_connection()
        if conn is None:
            return []

        conditions = []
        values = []

        if action:
            conditions.append("action = %s")
            values.append(action)
        if status:
            conditions.append("status = %s")
            values.append(status)
        if controller:
            conditions.append("controller = %s")
            values.append(controller)

        query = "SELECT * FROM runs"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY started_at DESC LIMIT %s"
        values.append(limit)

        with conn.cursor() as cur:
            cur.execute(query, values)
            rows = cur.fetchall()

        return [self._row_to_record(row) for row in rows]

    def delete_run(self, run_id: str) -> bool:
        conn = self._get_connection()
        if conn is None:
            return False

        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM artifacts WHERE run_id = %s", (run_id,))
                cur.execute("DELETE FROM runs WHERE run_id = %s", (run_id,))
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            return False

    def save_artifact(
        self,
        run_id: str,
        filename: str,
        data: Any,
    ) -> Optional[Path]:
        conn = self._get_connection()
        if conn is None:
            return None

        content = data if isinstance(data, str) else json.dumps(data, default=str)

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO artifacts (run_id, filename, content, created_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (run_id, filename)
                    DO UPDATE SET content = EXCLUDED.content, created_at = EXCLUDED.created_at
                    """,
                    (run_id, filename, content, datetime.now().isoformat()),
                )
            conn.commit()
            return Path(f"postgres://artifacts/{run_id}/{filename}")
        except Exception:
            conn.rollback()
            return None

    def get_artifact(self, run_id: str, filename: str) -> Optional[Any]:
        conn = self._get_connection()
        if conn is None:
            return None

        with conn.cursor() as cur:
            cur.execute(
                "SELECT content FROM artifacts WHERE run_id = %s AND filename = %s",
                (run_id, filename),
            )
            row = cur.fetchone()

        if row is None:
            return None

        content = row["content"]
        if filename.endswith(".json"):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass
        return content

    def _row_to_record(self, row) -> RunRecord:
        """Convert a database row to RunRecord."""

        def ensure_dict(val, default):
            if val is None:
                return default
            if isinstance(val, dict):
                return val
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except json.JSONDecodeError:
                    return default
            return default

        def ensure_list(val, default):
            if val is None:
                return default
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except json.JSONDecodeError:
                    return default
            return default

        started_at = row.get("started_at", "")
        if hasattr(started_at, "isoformat"):
            started_at = started_at.isoformat()

        completed_at = row.get("completed_at")
        if completed_at and hasattr(completed_at, "isoformat"):
            completed_at = completed_at.isoformat()

        return RunRecord(
            run_id=row["run_id"],
            name=row["name"],
            action=row["action"],
            status=row.get("status") or "unknown",
            started_at=started_at or "",
            completed_at=completed_at,
            duration_seconds=row.get("duration_seconds"),
            input_path=row.get("input_path") or "",
            output_path=row.get("output_path") or "",
            controller=row.get("controller") or "",
            parameters=ensure_dict(row.get("parameters"), {}),
            results=ensure_dict(row.get("results"), {}),
            output_files=ensure_list(row.get("output_files"), []),
            error_message=row.get("error_message"),
        )


def get_storage(
    backend: Optional[StorageBackend] = None,
    project_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
    connection_string: Optional[str] = None,
) -> BaseStorage:
    """
    Get a storage instance.

    Auto-detects backend from environment or project config if not specified.

    Args:
        backend: Storage backend to use (auto-detect if None)
        project_path: Project path for JSON storage
        db_path: Database path for SQLite storage
        connection_string: Connection string for PostgreSQL

    Returns:
        Storage instance
    """
    # Check environment variable
    if backend is None:
        env_backend = os.environ.get("BIOAMLA_STORAGE_BACKEND", "").lower()
        if env_backend == "sqlite":
            backend = StorageBackend.SQLITE
        elif env_backend == "postgres":
            backend = StorageBackend.POSTGRES
        elif env_backend == "json":
            backend = StorageBackend.JSON

    # Check for postgres connection string
    if backend is None and (connection_string or os.environ.get("BIOAMLA_DB_URL")):
        backend = StorageBackend.POSTGRES

    # Default to JSON
    if backend is None:
        backend = StorageBackend.JSON

    if backend == StorageBackend.JSON:
        return JSONStorage(project_path)
    elif backend == StorageBackend.SQLITE:
        return SQLiteStorage(db_path)
    elif backend == StorageBackend.POSTGRES:
        return PostgresStorage(connection_string)
    else:
        return JSONStorage(project_path)
