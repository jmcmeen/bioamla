"""
bioamla.workspace — project workspaces.

A workspace is a project directory with a ``.bioamla/`` folder holding its
metadata, settings, linked and imported artifacts (recordings, datasets, models),
annotations, run outputs and run history. Commands find the workspace by walking
up from the current directory, like git. See
[Workspaces, remote work and agent skills](../architecture/workspaces-and-remote.md)
for how workspaces fit with remote servers and agent skills.

Example:
    >>> from bioamla.workspace import create_workspace, find_workspace
    >>> ws = create_workspace("my-survey")
    >>> ws.link("/data/recordings/site-a")
    >>> find_workspace("my-survey/notes").name
    'my-survey'
"""

from bioamla.workspace._core import (
    ARTIFACT_KINDS,
    ARTIFACT_MODES,
    FORMAT_VERSION,
    HISTORY_RECORD_VERSION,
    WORKSPACE_DIR,
    Artifact,
    Workspace,
    create_workspace,
    current_workspace,
    find_workspace,
    is_workspace,
    open_workspace,
)

__all__ = [
    "ARTIFACT_KINDS",
    "ARTIFACT_MODES",
    "FORMAT_VERSION",
    "HISTORY_RECORD_VERSION",
    "WORKSPACE_DIR",
    "Artifact",
    "Workspace",
    "create_workspace",
    "current_workspace",
    "find_workspace",
    "is_workspace",
    "open_workspace",
]
