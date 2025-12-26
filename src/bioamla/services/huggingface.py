# services/huggingface.py
"""
Service for HuggingFace Hub model and dataset operations.
"""

from typing import Optional

from bioamla.models.huggingface import PushResult

from .base import BaseService, ServiceResult


class HuggingFaceService(BaseService):
    """
    Service for HuggingFace Hub operations.

    Provides high-level methods for:
    - Pushing models to the Hub
    - Pushing datasets to the Hub
    """

    def __init__(self) -> None:
        """Initialize HuggingFace service."""
        super().__init__()

    def _get_folder_size(self, path: str, limit: Optional[int] = None) -> int:
        """Calculate the total size of a folder in bytes."""
        import os

        total_size = 0
        for dirpath, _dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)
                    if limit is not None and total_size > limit:
                        return total_size
        return total_size

    def _count_files(self, path: str, limit: Optional[int] = None) -> int:
        """Count the total number of files in a folder."""
        import os

        count = 0
        for _dirpath, _dirnames, filenames in os.walk(path):
            count += len(filenames)
            if limit is not None and count > limit:
                return count
        return count

    def _is_large_folder(
        self,
        path: str,
        size_threshold_gb: float = 5.0,
        file_count_threshold: int = 1000,
    ) -> bool:
        """Determine if a folder should be uploaded using upload_large_folder."""
        size_threshold_bytes = int(size_threshold_gb * 1024 * 1024 * 1024)
        file_count = self._count_files(path, limit=file_count_threshold)
        if file_count > file_count_threshold:
            return True
        folder_size = self._get_folder_size(path, limit=size_threshold_bytes)
        return folder_size > size_threshold_bytes

    def push_model(
        self,
        path: str,
        repo_id: str,
        private: bool = False,
        commit_message: Optional[str] = None,
    ) -> ServiceResult[PushResult]:
        """
        Push a model folder to the HuggingFace Hub.

        Args:
            path: Local path to model folder
            repo_id: HuggingFace Hub repository ID
            private: Make the repository private
            commit_message: Custom commit message

        Returns:
            Result with push information
        """
        import os

        if not os.path.isdir(path):
            return ServiceResult.fail(f"Path '{path}' does not exist or is not a directory")

        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

            if self._is_large_folder(path):
                api.upload_large_folder(
                    folder_path=path,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=commit_message or "Upload model",
                )
            else:
                api.upload_folder(
                    folder_path=path,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=commit_message or "Upload model",
                )

            file_count = self._count_files(path)
            folder_size = self._get_folder_size(path)

            result = PushResult(
                repo_id=repo_id,
                repo_type="model",
                url=f"https://huggingface.co/{repo_id}",
                files_uploaded=file_count,
                total_size_bytes=folder_size,
            )

            return ServiceResult.ok(
                data=result,
                message=f"Successfully pushed model to {repo_id}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Push failed: {e}")

    def push_dataset(
        self,
        path: str,
        repo_id: str,
        private: bool = False,
        commit_message: Optional[str] = None,
    ) -> ServiceResult[PushResult]:
        """
        Push a dataset folder to the HuggingFace Hub.

        Args:
            path: Local path to dataset folder
            repo_id: HuggingFace Hub repository ID
            private: Make the repository private
            commit_message: Custom commit message

        Returns:
            Result with push information
        """
        import os

        if not os.path.isdir(path):
            return ServiceResult.fail(f"Path '{path}' does not exist or is not a directory")

        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

            if self._is_large_folder(path):
                api.upload_large_folder(
                    folder_path=path,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=commit_message or "Upload dataset",
                )
            else:
                api.upload_folder(
                    folder_path=path,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=commit_message or "Upload dataset",
                )

            file_count = self._count_files(path)
            folder_size = self._get_folder_size(path)

            result = PushResult(
                repo_id=repo_id,
                repo_type="dataset",
                url=f"https://huggingface.co/datasets/{repo_id}",
                files_uploaded=file_count,
                total_size_bytes=folder_size,
            )

            return ServiceResult.ok(
                data=result,
                message=f"Successfully pushed dataset to {repo_id}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Push failed: {e}")
