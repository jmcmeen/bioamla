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


def _tiny_audio_dataset(label_column="category", class_label=False):
    """Build a tiny in-memory audio Dataset (no network)."""
    import numpy as np

    from datasets import Audio, ClassLabel, Dataset, Features, Value

    n = 16000
    rows = {
        "audio": [
            {"array": np.zeros(n, dtype=np.float32), "sampling_rate": 16000, "path": "a.wav"},
            {"array": np.ones(n, dtype=np.float32) * 0.1, "sampling_rate": 16000, "path": "b.wav"},
            {"array": np.zeros(n, dtype=np.float32), "sampling_rate": 16000, "path": "c.wav"},
        ],
        label_column: ["dog", "cat", "dog"],
    }
    label_feature = ClassLabel(names=["cat", "dog"]) if class_label else Value("string")
    if class_label:
        rows[label_column] = [label_feature.str2int(v) for v in rows[label_column]]
    features = Features({"audio": Audio(sampling_rate=16000), label_column: label_feature})
    return Dataset.from_dict(rows, features=features)


class TestPullDataset:
    def test_bad_layout_raises(self, tmp_path) -> None:
        with pytest.raises(InvalidInputError):
            hf.pull_dataset("user/ds", str(tmp_path), layout="bogus")

    def test_materializes_audiofolder_and_metadata(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("datasets")
        import csv

        import datasets as hfds

        ds = _tiny_audio_dataset()
        monkeypatch.setattr(hfds, "load_dataset", lambda *a, **k: ds)

        result = hf.pull_dataset("user/ds", str(tmp_path / "out"), split="train")

        assert result.files_written == 3
        assert sorted(result.labels) == ["cat", "dog"]
        assert (tmp_path / "out" / "dog").is_dir()
        assert (tmp_path / "out" / "cat").is_dir()
        assert result.metadata_file is not None
        with open(result.metadata_file, newline="") as f:
            meta_rows = list(csv.DictReader(f))
        assert len(meta_rows) == 3
        assert {r["label"] for r in meta_rows} == {"cat", "dog"}
        # target is the sorted-label index
        assert {r["target"] for r in meta_rows} == {"0", "1"}

    def test_classlabel_decoded_to_names(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("datasets")
        import datasets as hfds

        ds = _tiny_audio_dataset(class_label=True)
        monkeypatch.setattr(hfds, "load_dataset", lambda *a, **k: ds)

        result = hf.pull_dataset("user/ds", str(tmp_path / "out"), split="train")
        assert sorted(result.labels) == ["cat", "dog"]

    def test_no_label_column_raises(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("datasets")
        import numpy as np

        import datasets as hfds
        from datasets import Audio, Dataset, Features

        ds = Dataset.from_dict(
            {
                "audio": [
                    {
                        "array": np.zeros(16000, dtype=np.float32),
                        "sampling_rate": 16000,
                        "path": "a.wav",
                    }
                ]
            },
            features=Features({"audio": Audio(sampling_rate=16000)}),
        )
        monkeypatch.setattr(hfds, "load_dataset", lambda *a, **k: ds)
        with pytest.raises(InvalidInputError):
            hf.pull_dataset("user/ds", str(tmp_path / "out"), split="train")
