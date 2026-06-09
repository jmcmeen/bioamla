"""HuggingFace Hub model and dataset publishing.

Push local model/dataset folders to the HuggingFace Hub. ``huggingface_hub`` is
a base dependency, imported lazily so importing this module stays light.

Upload failures raise :class:`~bioamla.exceptions.CatalogError`; a missing path
raises :class:`~bioamla.exceptions.InvalidInputError`.
"""

import logging
from pathlib import Path

from bioamla.catalogs._models import PushResult
from bioamla.exceptions import CatalogError, InvalidInputError

logger = logging.getLogger(__name__)


def _get_folder_size(path: str, limit: int | None = None) -> int:
    """Calculate the total size of a folder in bytes (optionally short-circuiting)."""
    total_size = 0
    for p in Path(path).glob("**/*"):
        if p.is_file():
            total_size += p.stat().st_size
            if limit is not None and total_size > limit:
                return total_size
    return total_size


def _count_files(path: str, limit: int | None = None) -> int:
    """Count the total number of files in a folder (optionally short-circuiting)."""
    count = 0
    for p in Path(path).glob("**/*"):
        if p.is_file():
            count += 1
            if limit is not None and count > limit:
                return count
    return count


def _is_large_folder(
    path: str,
    size_threshold_gb: float = 5.0,
    file_count_threshold: int = 1000,
) -> bool:
    """Return True if a folder should use ``upload_large_folder``."""
    size_threshold_bytes = int(size_threshold_gb * 1024 * 1024 * 1024)
    file_count = _count_files(path, limit=file_count_threshold)
    if file_count > file_count_threshold:
        return True
    folder_size = _get_folder_size(path, limit=size_threshold_bytes)
    return folder_size > size_threshold_bytes


def _get_hf_api():
    """Import and instantiate ``HfApi`` (lazy import)."""
    from huggingface_hub import HfApi

    return HfApi()


def _push_folder(
    path: str,
    repo_id: str,
    repo_type: str,
    private: bool,
    commit_message: str | None,
    url: str,
) -> PushResult:
    """Shared implementation for pushing a model or dataset folder."""
    if not Path(path).is_dir():
        raise InvalidInputError(f"Path '{path}' does not exist or is not a directory")

    api = _get_hf_api()
    default_message = f"Upload {repo_type}"
    try:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
        upload = api.upload_large_folder if _is_large_folder(path) else api.upload_folder
        upload(
            folder_path=path,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message or default_message,
        )
    except Exception as e:
        raise CatalogError(f"Push failed: {e}") from e

    return PushResult(
        repo_id=repo_id,
        repo_type=repo_type,
        url=url,
        files_uploaded=_count_files(path),
        total_size_bytes=_get_folder_size(path),
    )


def push_model(
    path: str,
    repo_id: str,
    private: bool = False,
    commit_message: str | None = None,
) -> PushResult:
    """Push a model folder to the HuggingFace Hub.

    Raises:
        InvalidInputError: if ``path`` is not a directory.
        CatalogError: on upload failure.
    """
    return _push_folder(
        path=path,
        repo_id=repo_id,
        repo_type="model",
        private=private,
        commit_message=commit_message,
        url=f"https://huggingface.co/{repo_id}",
    )


def push_dataset(
    path: str,
    repo_id: str,
    private: bool = False,
    commit_message: str | None = None,
) -> PushResult:
    """Push a dataset folder to the HuggingFace Hub.

    Raises:
        InvalidInputError: if ``path`` is not a directory.
        CatalogError: on upload failure.
    """
    return _push_folder(
        path=path,
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        commit_message=commit_message,
        url=f"https://huggingface.co/datasets/{repo_id}",
    )
