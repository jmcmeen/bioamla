"""
Project Management
==================

Handles bioamla project discovery, creation, and management.

A bioamla project is denoted by a `.bioamla/` directory marker which contains:
- config.toml: Project configuration
- models.toml: Registered models
- datasets.toml: Registered datasets
- logs/: Command history and logs
- runs/: Analysis run history and results
- cache/: Embeddings, model cache, temp files
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_MARKER = ".bioamla"
CONFIG_FILENAME = "config.toml"
MODELS_FILENAME = "models.toml"
DATASETS_FILENAME = "datasets.toml"
LOGS_DIR = "logs"
RUNS_DIR = "runs"
CACHE_DIR = "cache"

# Cache subdirectories
CACHE_EMBEDDINGS = "embeddings"
CACHE_MODELS = "models"
CACHE_TEMP = "temp"


@dataclass
class ProjectInfo:
    """Information about a bioamla project."""

    root: Path
    name: str
    version: str
    created: datetime
    description: str = ""

    # Paths (computed in __post_init__)
    config_path: Path = field(init=False)
    models_path: Path = field(init=False)
    datasets_path: Path = field(init=False)
    logs_path: Path = field(init=False)
    runs_path: Path = field(init=False)
    cache_path: Path = field(init=False)

    def __post_init__(self):
        project_dir = self.root / PROJECT_MARKER
        self.config_path = project_dir / CONFIG_FILENAME
        self.models_path = project_dir / MODELS_FILENAME
        self.datasets_path = project_dir / DATASETS_FILENAME
        self.logs_path = project_dir / LOGS_DIR
        self.runs_path = project_dir / RUNS_DIR
        self.cache_path = project_dir / CACHE_DIR

    @property
    def embeddings_cache(self) -> Path:
        """Path to embeddings cache directory."""
        return self.cache_path / CACHE_EMBEDDINGS

    @property
    def models_cache(self) -> Path:
        """Path to models cache directory."""
        return self.cache_path / CACHE_MODELS

    @property
    def temp_cache(self) -> Path:
        """Path to temporary files cache."""
        return self.cache_path / CACHE_TEMP


def find_project_root(start_path: Optional[Path] = None) -> Optional[Path]:
    """
    Find the project root by searching for .bioamla directory.

    Traverses from the start path upward through parent directories
    until a .bioamla directory is found or the filesystem root is reached.

    Args:
        start_path: Directory to start search from (default: cwd)

    Returns:
        Path to project root, or None if not in a project
    """
    if start_path is None:
        start_path = Path.cwd()

    current = start_path.resolve()

    while current != current.parent:
        if (current / PROJECT_MARKER).is_dir():
            return current
        current = current.parent

    # Check root directory
    if (current / PROJECT_MARKER).is_dir():
        return current

    return None


def is_in_project(path: Optional[Path] = None) -> bool:
    """
    Check if a path is within a bioamla project.

    Args:
        path: Path to check (default: cwd)

    Returns:
        True if the path is within a bioamla project
    """
    return find_project_root(path) is not None


def _get_template_content(template: str) -> str:
    """
    Get the content of a configuration template.

    Args:
        template: Template name ('default', 'minimal', 'research', 'production')

    Returns:
        TOML content as string

    Raises:
        ValueError: If template name is invalid
    """
    from importlib import resources

    try:
        # Python 3.9+
        files = resources.files("bioamla._internal.templates")
        template_file = files.joinpath(f"{template}.toml")
        return template_file.read_text()
    except (AttributeError, TypeError, FileNotFoundError):
        # Python 3.8 fallback or file not found
        try:
            return resources.read_text("bioamla._internal.templates", f"{template}.toml")
        except (FileNotFoundError, TypeError):
            pass

    # Final fallback - check relative path
    template_path = Path(__file__).parent / "templates" / f"{template}.toml"
    if template_path.exists():
        return template_path.read_text()

    raise ValueError(f"Unknown template: {template}")


def _customize_template(content: str, name: str, description: str = "") -> str:
    """
    Customize a template with project-specific values.

    Args:
        content: Template TOML content
        name: Project name
        description: Project description

    Returns:
        Customized TOML content
    """
    # Replace placeholders in template
    now = datetime.now(timezone.utc).isoformat()

    # Use simple string replacement for template values
    content = content.replace('name = ""', f'name = "{name}"')
    content = content.replace('description = ""', f'description = "{description}"')
    content = content.replace('created = ""', f'created = "{now}"')

    return content


def _create_models_toml(path: Path, project_name: str) -> None:
    """Create initial models.toml file."""
    content = f"""# Bioamla Models Registry - {project_name}
# Registered models for this project

# Default model for inference
[defaults]
inference = "MIT/ast-finetuned-audioset-10-10-0.4593"
embedding = ""

# Registered models
# Add models with: bioamla models register <name> <path_or_id>
#
# [models.<name>]
# id = "huggingface/model-id"  # or local path
# type = "ast"                  # ast, birdnet, custom
# description = "Model description"
# registered = "2024-01-01T00:00:00Z"
"""
    path.write_text(content)


def _create_datasets_toml(path: Path, project_name: str) -> None:
    """Create initial datasets.toml file."""
    content = f"""# Bioamla Datasets Registry - {project_name}
# Registered datasets for this project

# Default dataset paths
[defaults]
audio_dir = "audio"
output_dir = "output"

# Registered datasets
# Add datasets with: bioamla dataset register <name> <path>
#
# [datasets.<name>]
# path = "./audio/dataset_name"
# source = "inaturalist"        # inaturalist, xeno_canto, macaulay, local
# description = "Dataset description"
# registered = "2024-01-01T00:00:00Z"
# metadata = "metadata.csv"     # relative to dataset path
"""
    path.write_text(content)


def create_project(
    path: Path,
    name: Optional[str] = None,
    description: str = "",
    template: str = "default",
    config_file: Optional[Path] = None,
) -> ProjectInfo:
    """
    Create a new bioamla project.

    Creates a .bioamla directory with full project structure:
    - config.toml: Project configuration
    - models.toml: Model registry
    - datasets.toml: Dataset registry
    - logs/: Command history
    - runs/: Analysis results
    - cache/: Embeddings, models, temp files

    Args:
        path: Directory to create project in
        name: Project name (defaults to directory name)
        description: Project description
        template: Config template to use ('default', 'minimal', 'research', 'production')
        config_file: Optional custom config file to copy instead of template

    Returns:
        ProjectInfo for the created project

    Raises:
        FileNotFoundError: If path does not exist
        ValueError: If template is invalid
    """
    path = Path(path).resolve()

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    if name is None:
        name = path.name

    # Create .bioamla directory structure
    project_dir = path / PROJECT_MARKER
    project_dir.mkdir(exist_ok=True)

    # Create subdirectories
    (project_dir / LOGS_DIR).mkdir(exist_ok=True)
    (project_dir / RUNS_DIR).mkdir(exist_ok=True)
    (project_dir / CACHE_DIR).mkdir(exist_ok=True)
    (project_dir / CACHE_DIR / CACHE_EMBEDDINGS).mkdir(exist_ok=True)
    (project_dir / CACHE_DIR / CACHE_MODELS).mkdir(exist_ok=True)
    (project_dir / CACHE_DIR / CACHE_TEMP).mkdir(exist_ok=True)

    # Create config file
    config_path = project_dir / CONFIG_FILENAME

    if config_file is not None:
        # Copy provided config file
        import shutil

        shutil.copy(config_file, config_path)
    else:
        # Use template
        template_content = _get_template_content(template)
        customized_content = _customize_template(template_content, name, description)
        config_path.write_text(customized_content)

    # Create models.toml if it doesn't exist
    models_path = project_dir / MODELS_FILENAME
    if not models_path.exists():
        _create_models_toml(models_path, name)

    # Create datasets.toml if it doesn't exist
    datasets_path = project_dir / DATASETS_FILENAME
    if not datasets_path.exists():
        _create_datasets_toml(datasets_path, name)

    # Create project info
    now = datetime.now(timezone.utc)

    return ProjectInfo(
        root=path,
        name=name,
        version="1.0.0",
        created=now,
        description=description,
    )


def load_project(path: Optional[Path] = None) -> Optional[ProjectInfo]:
    """
    Load project information from a path.

    Args:
        path: Path within project (default: cwd)

    Returns:
        ProjectInfo if in a project, None otherwise
    """
    project_root = find_project_root(path)

    if project_root is None:
        return None

    config_path = project_root / PROJECT_MARKER / CONFIG_FILENAME

    if not config_path.exists():
        # Project marker exists but no config - create minimal info
        return ProjectInfo(
            root=project_root,
            name=project_root.name,
            version="1.0.0",
            created=datetime.now(timezone.utc),
        )

    # Load project info from config
    from bioamla.core.config import load_toml

    try:
        config_data = load_toml(config_path)
        project_section = config_data.get("project", {})

        # Parse created timestamp
        created_str = project_section.get("created", "")
        if created_str:
            try:
                created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            except ValueError:
                created = datetime.now(timezone.utc)
        else:
            created = datetime.now(timezone.utc)

        return ProjectInfo(
            root=project_root,
            name=project_section.get("name", project_root.name),
            version=project_section.get("version", "1.0.0"),
            created=created,
            description=project_section.get("description", ""),
        )
    except Exception:
        # Config exists but couldn't be parsed
        return ProjectInfo(
            root=project_root,
            name=project_root.name,
            version="1.0.0",
            created=datetime.now(timezone.utc),
        )


def get_project_config_path(path: Optional[Path] = None) -> Optional[Path]:
    """
    Get the path to the project configuration file.

    Args:
        path: Path within project (default: cwd)

    Returns:
        Path to config.toml if in a project, None otherwise
    """
    project_root = find_project_root(path)

    if project_root is None:
        return None

    config_path = project_root / PROJECT_MARKER / CONFIG_FILENAME

    if config_path.exists():
        return config_path

    return None
