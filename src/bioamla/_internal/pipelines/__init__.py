# _internal/pipelines/__init__.py
"""
Built-in pipeline templates for common bioacoustics tasks.

Available templates:
- bird_detection.toml: Bird species detection pipeline
- amphibian_detection.toml: Amphibian call detection pipeline
- indices_batch.toml: Acoustic indices batch analysis
- embedding_clustering.toml: Embedding extraction and clustering
- audio_preprocessing.toml: Audio preprocessing pipeline
"""

from pathlib import Path

PIPELINES_DIR = Path(__file__).parent


def get_pipeline_template(name: str) -> str:
    """Get pipeline template content by name.

    Args:
        name: Template name (with or without .toml extension)

    Returns:
        Template content as string

    Raises:
        FileNotFoundError: If template does not exist
    """
    if not name.endswith(".toml"):
        name = f"{name}.toml"

    path = PIPELINES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Pipeline template not found: {name}")

    return path.read_text()


def list_pipeline_templates() -> list[dict]:
    """List available pipeline templates.

    Returns:
        List of dicts with name and description for each template
    """
    templates = []
    for path in PIPELINES_DIR.glob("*.toml"):
        content = path.read_text()
        # Extract description from first comment or pipeline.description
        description = ""
        for line in content.split("\n"):
            if line.startswith("# "):
                description = line[2:].strip()
                break
            if "description" in line and "=" in line:
                description = line.split("=", 1)[1].strip().strip('"')
                break

        templates.append(
            {
                "name": path.stem,
                "file": path.name,
                "description": description,
            }
        )

    return sorted(templates, key=lambda x: x["name"])
