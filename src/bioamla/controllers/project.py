# controllers/project.py
"""
Project Controller
==================

Controller for bioamla project management.

Provides a clean interface for:
- Creating and initializing projects
- Loading project information
- Managing project configuration
- Project status and statistics

Usage:
    from bioamla.controllers import ProjectController

    controller = ProjectController()

    # Create a new project
    result = controller.create("./my-project", name="Frog Study")
    if result.success:
        print(f"Created: {result.data.name}")

    # Load current project
    result = controller.load()
    if result.success:
        print(f"Project: {result.data.name}")

    # Get project statistics
    result = controller.get_stats()
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseController, ControllerResult, ToDictMixin


@dataclass
class ProjectSummary(ToDictMixin):
    """Summary of a bioamla project."""

    name: str
    root: str
    version: str
    created: str
    description: str
    config_path: str
    models_path: str
    datasets_path: str
    logs_path: str
    runs_path: str
    cache_path: str


@dataclass
class ProjectStatistics(ToDictMixin):
    """Statistics about a project."""

    name: str
    root: str
    audio_files: int = 0
    total_size_mb: float = 0.0
    datasets: List[str] = field(default_factory=list)
    registered_models: int = 0
    registered_datasets: int = 0
    run_count: int = 0
    cache_size_mb: float = 0.0
    has_metadata: bool = False
    command_count: int = 0
    last_command: Optional[str] = None


@dataclass
class ConfigSummary(ToDictMixin):
    """Summary of project configuration."""

    path: str
    sections: List[str]
    settings: Dict[str, Any]


@dataclass
class ModelInfo(ToDictMixin):
    """Information about a registered model."""

    name: str
    model_id: str
    model_type: str = "ast"
    description: str = ""
    registered: str = ""


@dataclass
class DatasetInfo(ToDictMixin):
    """Information about a registered dataset."""

    name: str
    path: str
    source: str = "local"
    description: str = ""
    registered: str = ""
    metadata_file: str = "metadata.csv"


@dataclass
class RunInfo(ToDictMixin):
    """Information about an analysis run."""

    run_id: str
    name: str
    started: str
    completed: Optional[str] = None
    status: str = "running"
    action: str = ""
    input_path: str = ""
    output_path: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)


class ProjectController(BaseController):
    """
    Controller for bioamla project management.

    Handles project creation, loading, configuration, and statistics.
    """

    def __init__(self):
        super().__init__()

    # =========================================================================
    # Project Creation and Loading
    # =========================================================================

    def create(
        self,
        path: str,
        name: Optional[str] = None,
        description: str = "",
        template: str = "default",
        config_file: Optional[str] = None,
        force: bool = False,
    ) -> ControllerResult[ProjectSummary]:
        """
        Create a new bioamla project.

        Args:
            path: Directory to create project in
            name: Project name (defaults to directory name)
            description: Project description
            template: Config template ('default', 'minimal', 'research', 'production')
            config_file: Optional custom config file to use
            force: Overwrite existing project if True

        Returns:
            ControllerResult containing ProjectSummary
        """
        from bioamla.core.project import PROJECT_MARKER, create_project

        try:
            project_path = Path(path).resolve()

            # Check for existing project
            if (project_path / PROJECT_MARKER).exists() and not force:
                return ControllerResult.fail(
                    f"Project already exists at {project_path}. Use force=True to reinitialize."
                )

            # Create the project
            info = create_project(
                path=project_path,
                name=name,
                description=description,
                template=template,
                config_file=Path(config_file) if config_file else None,
            )

            summary = ProjectSummary(
                name=info.name,
                root=str(info.root),
                version=info.version,
                created=info.created.isoformat(),
                description=info.description,
                config_path=str(info.config_path),
                models_path=str(info.models_path),
                datasets_path=str(info.datasets_path),
                logs_path=str(info.logs_path),
                runs_path=str(info.runs_path),
                cache_path=str(info.cache_path),
            )

            return ControllerResult.ok(
                data=summary,
                message=f"Created project '{info.name}' at {info.root}",
            )

        except ValueError as e:
            return ControllerResult.fail(f"Invalid template: {e}")
        except Exception as e:
            return ControllerResult.fail(f"Failed to create project: {e}")

    def load(self, path: Optional[str] = None) -> ControllerResult[ProjectSummary]:
        """
        Load project information from a path.

        Args:
            path: Path within project (default: current directory)

        Returns:
            ControllerResult containing ProjectSummary
        """
        from bioamla.core.project import load_project

        try:
            info = load_project(Path(path) if path else None)

            if info is None:
                return ControllerResult.fail("Not in a bioamla project")

            summary = ProjectSummary(
                name=info.name,
                root=str(info.root),
                version=info.version,
                created=info.created.isoformat(),
                description=info.description,
                config_path=str(info.config_path),
                models_path=str(info.models_path),
                datasets_path=str(info.datasets_path),
                logs_path=str(info.logs_path),
                runs_path=str(info.runs_path),
                cache_path=str(info.cache_path),
            )

            return ControllerResult.ok(
                data=summary,
                message=f"Loaded project: {info.name}",
            )

        except Exception as e:
            return ControllerResult.fail(f"Failed to load project: {e}")

    def exists(self, path: Optional[str] = None) -> ControllerResult[bool]:
        """
        Check if a path is within a bioamla project.

        Args:
            path: Path to check (default: current directory)

        Returns:
            ControllerResult containing boolean
        """
        from bioamla.core.project import is_in_project

        try:
            in_project = is_in_project(Path(path) if path else None)
            return ControllerResult.ok(
                data=in_project,
                message="In project" if in_project else "Not in project",
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to check project: {e}")

    def find_root(self, path: Optional[str] = None) -> ControllerResult[str]:
        """
        Find the project root directory.

        Args:
            path: Starting path for search (default: current directory)

        Returns:
            ControllerResult containing root path string
        """
        from bioamla.core.project import find_project_root

        try:
            root = find_project_root(Path(path) if path else None)

            if root is None:
                return ControllerResult.fail("Not in a bioamla project")

            return ControllerResult.ok(
                data=str(root),
                message=f"Project root: {root}",
            )
        except Exception as e:
            return ControllerResult.fail(f"Failed to find project root: {e}")

    # =========================================================================
    # Project Configuration
    # =========================================================================

    def get_config(self, path: Optional[str] = None) -> ControllerResult[ConfigSummary]:
        """
        Get project configuration.

        Args:
            path: Path within project (default: current directory)

        Returns:
            ControllerResult containing ConfigSummary
        """
        from bioamla.core.config import load_toml
        from bioamla.core.project import get_project_config_path

        try:
            config_path = get_project_config_path(Path(path) if path else None)

            if config_path is None:
                return ControllerResult.fail("Not in a bioamla project or no config found")

            config_data = load_toml(config_path)

            summary = ConfigSummary(
                path=str(config_path),
                sections=list(config_data.keys()),
                settings=config_data,
            )

            return ControllerResult.ok(
                data=summary,
                message=f"Loaded config from {config_path}",
            )

        except Exception as e:
            return ControllerResult.fail(f"Failed to load config: {e}")

    def update_config(
        self,
        updates: Dict[str, Any],
        path: Optional[str] = None,
    ) -> ControllerResult[ConfigSummary]:
        """
        Update project configuration.

        Args:
            updates: Dictionary of config updates (nested keys supported)
            path: Path within project (default: current directory)

        Returns:
            ControllerResult containing updated ConfigSummary
        """
        from bioamla.core.config import load_toml, save_toml
        from bioamla.core.project import get_project_config_path

        try:
            config_path = get_project_config_path(Path(path) if path else None)

            if config_path is None:
                return ControllerResult.fail("Not in a bioamla project or no config found")

            # Load existing config
            config_data = load_toml(config_path)

            # Apply updates (deep merge)
            def deep_update(base: dict, updates: dict) -> dict:
                for key, value in updates.items():
                    if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                        deep_update(base[key], value)
                    else:
                        base[key] = value
                return base

            deep_update(config_data, updates)

            # Save updated config
            save_toml(config_path, config_data)

            summary = ConfigSummary(
                path=str(config_path),
                sections=list(config_data.keys()),
                settings=config_data,
            )

            return ControllerResult.ok(
                data=summary,
                message=f"Updated config at {config_path}",
            )

        except Exception as e:
            return ControllerResult.fail(f"Failed to update config: {e}")

    def reset_config(
        self,
        template: str = "default",
        path: Optional[str] = None,
    ) -> ControllerResult[ConfigSummary]:
        """
        Reset project configuration to template defaults.

        Args:
            template: Template to reset to ('default', 'minimal', 'research', 'production')
            path: Path within project (default: current directory)

        Returns:
            ControllerResult containing reset ConfigSummary
        """
        from bioamla.core.project import (
            _customize_template,
            _get_template_content,
            load_project,
        )

        try:
            info = load_project(Path(path) if path else None)

            if info is None:
                return ControllerResult.fail("Not in a bioamla project")

            # Get template and customize
            template_content = _get_template_content(template)
            customized = _customize_template(template_content, info.name, info.description)

            # Write to config
            info.config_path.write_text(customized)

            # Return updated config
            return self.get_config(path)

        except ValueError as e:
            return ControllerResult.fail(f"Invalid template: {e}")
        except Exception as e:
            return ControllerResult.fail(f"Failed to reset config: {e}")

    # =========================================================================
    # Project Statistics
    # =========================================================================

    def get_stats(self, path: Optional[str] = None) -> ControllerResult[ProjectStatistics]:
        """
        Get project statistics.

        Args:
            path: Path within project (default: current directory)

        Returns:
            ControllerResult containing ProjectStats
        """
        from bioamla.core.globals import SUPPORTED_AUDIO_EXTENSIONS
        from bioamla.core.log import CommandLogger
        from bioamla.core.project import load_project

        try:
            info = load_project(Path(path) if path else None)

            if info is None:
                return ControllerResult.fail("Not in a bioamla project")

            # Count audio files
            audio_count = 0
            total_size = 0
            datasets = []

            for ext in SUPPORTED_AUDIO_EXTENSIONS:
                for audio_file in info.root.rglob(f"*{ext}"):
                    audio_count += 1
                    total_size += audio_file.stat().st_size

            # Find datasets (directories with metadata.csv)
            for metadata_file in info.root.rglob("metadata.csv"):
                datasets.append(metadata_file.parent.name)

            # Check for metadata.csv in root
            has_metadata = (info.root / "metadata.csv").exists()

            # Get command history stats
            command_logger = CommandLogger(info.root)
            command_stats = command_logger.get_stats()
            command_count = command_stats.get("total_commands", 0)

            # Get last command
            history = command_logger.get_history(limit=1)
            last_command = history[0].command if history else None

            stats = ProjectStatistics(
                name=info.name,
                root=str(info.root),
                audio_files=audio_count,
                total_size_mb=round(total_size / (1024 * 1024), 2),
                datasets=datasets,
                has_metadata=has_metadata,
                command_count=command_count,
                last_command=last_command,
            )

            return ControllerResult.ok(
                data=stats,
                message=f"Project stats for {info.name}",
            )

        except Exception as e:
            return ControllerResult.fail(f"Failed to get project stats: {e}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def list_templates(self) -> ControllerResult[List[str]]:
        """
        List available project templates.

        Returns:
            ControllerResult containing list of template names
        """
        templates = ["default", "minimal", "research", "production"]
        return ControllerResult.ok(
            data=templates,
            message=f"{len(templates)} templates available",
        )

    def describe_template(self, template: str) -> ControllerResult[Dict[str, str]]:
        """
        Get description of a project template.

        Args:
            template: Template name

        Returns:
            ControllerResult containing template description
        """
        descriptions = {
            "default": "Balanced settings for general use. Good starting point for most projects.",
            "minimal": "Minimal configuration with most values using defaults. Fast startup.",
            "research": "Detailed logging and reproducibility focused. Ideal for experiments.",
            "production": "Optimized for batch processing and performance.",
        }

        if template not in descriptions:
            return ControllerResult.fail(
                f"Unknown template: {template}. Available: {list(descriptions.keys())}"
            )

        return ControllerResult.ok(
            data={"name": template, "description": descriptions[template]},
            message=f"Template: {template}",
        )

    # =========================================================================
    # Model Registry
    # =========================================================================

    def register_model(
        self,
        name: str,
        model_id: str,
        model_type: str = "ast",
        description: str = "",
        path: Optional[str] = None,
    ) -> ControllerResult[ModelInfo]:
        """
        Register a model with the project.

        Args:
            name: Short name for the model
            model_id: HuggingFace model ID or local path
            model_type: Model type (ast, birdnet, custom)
            description: Model description
            path: Path within project (default: current directory)

        Returns:
            ControllerResult containing ModelInfo
        """
        from bioamla.core.config import load_toml, save_toml
        from bioamla.core.project import load_project

        try:
            info = load_project(Path(path) if path else None)
            if info is None:
                return ControllerResult.fail("Not in a bioamla project")

            # Load existing models.toml
            models_data = {}
            if info.models_path.exists():
                models_data = load_toml(info.models_path)

            # Add model entry
            if "models" not in models_data:
                models_data["models"] = {}

            now = datetime.now().isoformat()
            models_data["models"][name] = {
                "id": model_id,
                "type": model_type,
                "description": description,
                "registered": now,
            }

            # Save
            save_toml(models_data, info.models_path)

            model_info = ModelInfo(
                name=name,
                model_id=model_id,
                model_type=model_type,
                description=description,
                registered=now,
            )

            return ControllerResult.ok(
                data=model_info,
                message=f"Registered model '{name}'",
            )

        except Exception as e:
            return ControllerResult.fail(f"Failed to register model: {e}")

    def list_models(self, path: Optional[str] = None) -> ControllerResult[List[ModelInfo]]:
        """
        List registered models.

        Args:
            path: Path within project (default: current directory)

        Returns:
            ControllerResult containing list of ModelInfo
        """
        from bioamla.core.config import load_toml
        from bioamla.core.project import load_project

        try:
            info = load_project(Path(path) if path else None)
            if info is None:
                return ControllerResult.fail("Not in a bioamla project")

            if not info.models_path.exists():
                return ControllerResult.ok(data=[], message="No models registered")

            models_data = load_toml(info.models_path)
            models = []

            for name, data in models_data.get("models", {}).items():
                models.append(
                    ModelInfo(
                        name=name,
                        model_id=data.get("id", ""),
                        model_type=data.get("type", "ast"),
                        description=data.get("description", ""),
                        registered=data.get("registered", ""),
                    )
                )

            return ControllerResult.ok(
                data=models,
                message=f"{len(models)} models registered",
            )

        except Exception as e:
            return ControllerResult.fail(f"Failed to list models: {e}")

    # =========================================================================
    # Dataset Registry
    # =========================================================================

    def register_dataset(
        self,
        name: str,
        dataset_path: str,
        source: str = "local",
        description: str = "",
        metadata_file: str = "metadata.csv",
        path: Optional[str] = None,
    ) -> ControllerResult[DatasetInfo]:
        """
        Register a dataset with the project.

        Args:
            name: Short name for the dataset
            dataset_path: Path to dataset directory
            source: Data source (local, inaturalist, xeno_canto, macaulay)
            description: Dataset description
            metadata_file: Name of metadata file
            path: Path within project (default: current directory)

        Returns:
            ControllerResult containing DatasetInfo
        """
        from bioamla.core.config import load_toml, save_toml
        from bioamla.core.project import load_project

        try:
            info = load_project(Path(path) if path else None)
            if info is None:
                return ControllerResult.fail("Not in a bioamla project")

            # Load existing datasets.toml
            datasets_data = {}
            if info.datasets_path.exists():
                datasets_data = load_toml(info.datasets_path)

            # Add dataset entry
            if "datasets" not in datasets_data:
                datasets_data["datasets"] = {}

            now = datetime.now().isoformat()
            datasets_data["datasets"][name] = {
                "path": dataset_path,
                "source": source,
                "description": description,
                "registered": now,
                "metadata": metadata_file,
            }

            # Save
            save_toml(datasets_data, info.datasets_path)

            dataset_info = DatasetInfo(
                name=name,
                path=dataset_path,
                source=source,
                description=description,
                registered=now,
                metadata_file=metadata_file,
            )

            return ControllerResult.ok(
                data=dataset_info,
                message=f"Registered dataset '{name}'",
            )

        except Exception as e:
            return ControllerResult.fail(f"Failed to register dataset: {e}")

    def list_datasets(self, path: Optional[str] = None) -> ControllerResult[List[DatasetInfo]]:
        """
        List registered datasets.

        Args:
            path: Path within project (default: current directory)

        Returns:
            ControllerResult containing list of DatasetInfo
        """
        from bioamla.core.config import load_toml
        from bioamla.core.project import load_project

        try:
            info = load_project(Path(path) if path else None)
            if info is None:
                return ControllerResult.fail("Not in a bioamla project")

            if not info.datasets_path.exists():
                return ControllerResult.ok(data=[], message="No datasets registered")

            datasets_data = load_toml(info.datasets_path)
            datasets = []

            for name, data in datasets_data.get("datasets", {}).items():
                datasets.append(
                    DatasetInfo(
                        name=name,
                        path=data.get("path", ""),
                        source=data.get("source", "local"),
                        description=data.get("description", ""),
                        registered=data.get("registered", ""),
                        metadata_file=data.get("metadata", "metadata.csv"),
                    )
                )

            return ControllerResult.ok(
                data=datasets,
                message=f"{len(datasets)} datasets registered",
            )

        except Exception as e:
            return ControllerResult.fail(f"Failed to list datasets: {e}")

    # =========================================================================
    # Run Management
    # =========================================================================

    def create_run(
        self,
        name: str,
        action: str,
        input_path: str = "",
        output_path: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        path: Optional[str] = None,
    ) -> ControllerResult[RunInfo]:
        """
        Create a new analysis run.

        Args:
            name: Run name/description
            action: Action being performed (e.g., 'predict', 'embed', 'indices')
            input_path: Input file/directory
            output_path: Output file/directory
            parameters: Run parameters
            path: Path within project (default: current directory)

        Returns:
            ControllerResult containing RunInfo
        """
        import json
        import uuid

        from bioamla.core.project import load_project

        try:
            info = load_project(Path(path) if path else None)
            if info is None:
                return ControllerResult.fail("Not in a bioamla project")

            # Generate run ID
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

            # Create run directory
            run_dir = info.runs_path / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            now = datetime.now().isoformat()
            run_info = RunInfo(
                run_id=run_id,
                name=name,
                started=now,
                status="running",
                action=action,
                input_path=input_path,
                output_path=output_path,
                parameters=parameters or {},
            )

            # Save run metadata
            metadata_path = run_dir / "run.json"
            metadata_path.write_text(json.dumps(run_info.to_dict(), indent=2))

            return ControllerResult.ok(
                data=run_info,
                message=f"Created run '{run_id}'",
            )

        except Exception as e:
            return ControllerResult.fail(f"Failed to create run: {e}")

    def complete_run(
        self,
        run_id: str,
        status: str = "completed",
        results: Optional[Dict[str, Any]] = None,
        path: Optional[str] = None,
    ) -> ControllerResult[RunInfo]:
        """
        Mark a run as complete.

        Args:
            run_id: Run ID to complete
            status: Final status (completed, failed, cancelled)
            results: Run results/summary
            path: Path within project (default: current directory)

        Returns:
            ControllerResult containing updated RunInfo
        """
        import json

        from bioamla.core.project import load_project

        try:
            info = load_project(Path(path) if path else None)
            if info is None:
                return ControllerResult.fail("Not in a bioamla project")

            run_dir = info.runs_path / run_id
            metadata_path = run_dir / "run.json"

            if not metadata_path.exists():
                return ControllerResult.fail(f"Run not found: {run_id}")

            # Load existing metadata
            run_data = json.loads(metadata_path.read_text())

            # Update
            run_data["completed"] = datetime.now().isoformat()
            run_data["status"] = status
            if results:
                run_data["results"] = results

            # Save
            metadata_path.write_text(json.dumps(run_data, indent=2))

            run_info = RunInfo(
                run_id=run_data["run_id"],
                name=run_data["name"],
                started=run_data["started"],
                completed=run_data.get("completed"),
                status=run_data["status"],
                action=run_data.get("action", ""),
                input_path=run_data.get("input_path", ""),
                output_path=run_data.get("output_path", ""),
                parameters=run_data.get("parameters", {}),
                results=run_data.get("results", {}),
            )

            return ControllerResult.ok(
                data=run_info,
                message=f"Run '{run_id}' marked as {status}",
            )

        except Exception as e:
            return ControllerResult.fail(f"Failed to complete run: {e}")

    def list_runs(
        self,
        limit: int = 10,
        status: Optional[str] = None,
        path: Optional[str] = None,
    ) -> ControllerResult[List[RunInfo]]:
        """
        List analysis runs.

        Args:
            limit: Maximum number of runs to return
            status: Filter by status (running, completed, failed)
            path: Path within project (default: current directory)

        Returns:
            ControllerResult containing list of RunInfo
        """
        import json

        from bioamla.core.project import load_project

        try:
            info = load_project(Path(path) if path else None)
            if info is None:
                return ControllerResult.fail("Not in a bioamla project")

            if not info.runs_path.exists():
                return ControllerResult.ok(data=[], message="No runs found")

            runs = []
            for run_dir in sorted(info.runs_path.iterdir(), reverse=True):
                if not run_dir.is_dir():
                    continue

                metadata_path = run_dir / "run.json"
                if not metadata_path.exists():
                    continue

                run_data = json.loads(metadata_path.read_text())

                # Filter by status
                if status and run_data.get("status") != status:
                    continue

                runs.append(
                    RunInfo(
                        run_id=run_data["run_id"],
                        name=run_data["name"],
                        started=run_data["started"],
                        completed=run_data.get("completed"),
                        status=run_data.get("status", "unknown"),
                        action=run_data.get("action", ""),
                        input_path=run_data.get("input_path", ""),
                        output_path=run_data.get("output_path", ""),
                        parameters=run_data.get("parameters", {}),
                        results=run_data.get("results", {}),
                    )
                )

                if len(runs) >= limit:
                    break

            return ControllerResult.ok(
                data=runs,
                message=f"{len(runs)} runs found",
            )

        except Exception as e:
            return ControllerResult.fail(f"Failed to list runs: {e}")

    def get_run(self, run_id: str, path: Optional[str] = None) -> ControllerResult[RunInfo]:
        """
        Get details for a specific run.

        Args:
            run_id: Run ID
            path: Path within project (default: current directory)

        Returns:
            ControllerResult containing RunInfo
        """
        import json

        from bioamla.core.project import load_project

        try:
            info = load_project(Path(path) if path else None)
            if info is None:
                return ControllerResult.fail("Not in a bioamla project")

            run_dir = info.runs_path / run_id
            metadata_path = run_dir / "run.json"

            if not metadata_path.exists():
                return ControllerResult.fail(f"Run not found: {run_id}")

            run_data = json.loads(metadata_path.read_text())

            run_info = RunInfo(
                run_id=run_data["run_id"],
                name=run_data["name"],
                started=run_data["started"],
                completed=run_data.get("completed"),
                status=run_data.get("status", "unknown"),
                action=run_data.get("action", ""),
                input_path=run_data.get("input_path", ""),
                output_path=run_data.get("output_path", ""),
                parameters=run_data.get("parameters", {}),
                results=run_data.get("results", {}),
            )

            return ControllerResult.ok(
                data=run_info,
                message=f"Run: {run_id}",
            )

        except Exception as e:
            return ControllerResult.fail(f"Failed to get run: {e}")

    # =========================================================================
    # Cache Management
    # =========================================================================

    def get_cache_stats(self, path: Optional[str] = None) -> ControllerResult[Dict[str, Any]]:
        """
        Get cache statistics.

        Args:
            path: Path within project (default: current directory)

        Returns:
            ControllerResult containing cache statistics
        """
        from bioamla.core.project import load_project

        try:
            info = load_project(Path(path) if path else None)
            if info is None:
                return ControllerResult.fail("Not in a bioamla project")

            def get_dir_size(dir_path: Path) -> tuple:
                """Get directory size and file count."""
                total_size = 0
                file_count = 0
                if dir_path.exists():
                    for f in dir_path.rglob("*"):
                        if f.is_file():
                            total_size += f.stat().st_size
                            file_count += 1
                return total_size, file_count

            embeddings_size, embeddings_count = get_dir_size(info.embeddings_cache)
            models_size, models_count = get_dir_size(info.models_cache)
            temp_size, temp_count = get_dir_size(info.temp_cache)

            total_size = embeddings_size + models_size + temp_size

            stats = {
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "embeddings": {
                    "path": str(info.embeddings_cache),
                    "size_mb": round(embeddings_size / (1024 * 1024), 2),
                    "files": embeddings_count,
                },
                "models": {
                    "path": str(info.models_cache),
                    "size_mb": round(models_size / (1024 * 1024), 2),
                    "files": models_count,
                },
                "temp": {
                    "path": str(info.temp_cache),
                    "size_mb": round(temp_size / (1024 * 1024), 2),
                    "files": temp_count,
                },
            }

            return ControllerResult.ok(
                data=stats,
                message=f"Cache size: {stats['total_size_mb']} MB",
            )

        except Exception as e:
            return ControllerResult.fail(f"Failed to get cache stats: {e}")

    def clear_cache(
        self,
        cache_type: Optional[str] = None,
        path: Optional[str] = None,
    ) -> ControllerResult[Dict[str, int]]:
        """
        Clear project cache.

        Args:
            cache_type: Type to clear (embeddings, models, temp, or None for all)
            path: Path within project (default: current directory)

        Returns:
            ControllerResult containing count of deleted files
        """
        import shutil

        from bioamla.core.project import load_project

        try:
            info = load_project(Path(path) if path else None)
            if info is None:
                return ControllerResult.fail("Not in a bioamla project")

            deleted = {"embeddings": 0, "models": 0, "temp": 0}

            def clear_dir(dir_path: Path, cache_name: str) -> int:
                count = 0
                if dir_path.exists():
                    for f in dir_path.iterdir():
                        if f.is_file():
                            f.unlink()
                            count += 1
                        elif f.is_dir():
                            shutil.rmtree(f)
                            count += 1
                return count

            if cache_type is None or cache_type == "embeddings":
                deleted["embeddings"] = clear_dir(info.embeddings_cache, "embeddings")

            if cache_type is None or cache_type == "models":
                deleted["models"] = clear_dir(info.models_cache, "models")

            if cache_type is None or cache_type == "temp":
                deleted["temp"] = clear_dir(info.temp_cache, "temp")

            total = sum(deleted.values())

            return ControllerResult.ok(
                data=deleted,
                message=f"Cleared {total} items from cache",
            )

        except Exception as e:
            return ControllerResult.fail(f"Failed to clear cache: {e}")
