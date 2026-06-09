"""Tests for HuggingFace catalog (path validation, helpers, errors; no network)."""

import pytest

from bioamla.catalogs import huggingface as hf
from bioamla.exceptions import CatalogError, InvalidInputError


class TestFolderHelpers:
    def test_count_and_size(self, tmp_path) -> None:
        (tmp_path / "a.txt").write_text("hello")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.txt").write_text("world!!")
        assert hf._count_files(str(tmp_path)) == 2
        assert hf._get_folder_size(str(tmp_path)) == len("hello") + len("world!!")

    def test_is_large_folder_false_for_small(self, tmp_path) -> None:
        (tmp_path / "a.txt").write_text("x")
        assert hf._is_large_folder(str(tmp_path)) is False


class TestPathValidation:
    def test_push_model_missing_path_raises(self, tmp_path) -> None:
        with pytest.raises(InvalidInputError):
            hf.push_model(str(tmp_path / "nope"), "user/repo")

    def test_push_dataset_missing_path_raises(self, tmp_path) -> None:
        with pytest.raises(InvalidInputError):
            hf.push_dataset(str(tmp_path / "nope"), "user/repo")


class TestUploadFailure:
    def test_upload_failure_raises_catalog_error(self, tmp_path, monkeypatch) -> None:
        (tmp_path / "model.bin").write_text("x")

        class FakeApi:
            def create_repo(self, **kwargs):
                pass

            def upload_folder(self, **kwargs):
                raise RuntimeError("upload boom")

            def upload_large_folder(self, **kwargs):
                raise RuntimeError("upload boom")

        monkeypatch.setattr(hf, "_get_hf_api", lambda: FakeApi())
        with pytest.raises(CatalogError):
            hf.push_model(str(tmp_path), "user/repo")

    def test_push_model_success(self, tmp_path, monkeypatch) -> None:
        (tmp_path / "model.bin").write_text("xyz")

        class FakeApi:
            def create_repo(self, **kwargs):
                pass

            def upload_folder(self, **kwargs):
                pass

        monkeypatch.setattr(hf, "_get_hf_api", lambda: FakeApi())
        result = hf.push_model(str(tmp_path), "user/repo")
        assert result.repo_id == "user/repo"
        assert result.repo_type == "model"
        assert result.url == "https://huggingface.co/user/repo"
        assert result.files_uploaded == 1
