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


class _FakeRepo:
    def __init__(self, repo_id, repo_type, size):
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.size_on_disk = size
        self.revisions = []


class _FakeCache:
    def __init__(self, repos):
        self.repos = repos


class TestCache:
    def _patch(self, monkeypatch):
        import huggingface_hub

        repos = [
            _FakeRepo("user/model-a", "model", 100),
            _FakeRepo("user/ds-a", "dataset", 200),
            _FakeRepo("user/ds-b", "dataset", 50),
        ]
        monkeypatch.setattr(huggingface_hub, "scan_cache_dir", lambda: _FakeCache(repos))

    def test_scan_filters_by_type(self, monkeypatch) -> None:
        self._patch(monkeypatch)
        assert {r.repo_id for r in hf.scan_cache(models=False, datasets=True)} == {
            "user/ds-a",
            "user/ds-b",
        }
        assert {r.repo_id for r in hf.scan_cache(models=True, datasets=False)} == {"user/model-a"}

    def test_purge_counts_and_frees(self, monkeypatch) -> None:
        self._patch(monkeypatch)
        # Fake repos have no on-disk paths, so rmtree(ignore_errors) is a safe no-op.
        result = hf.purge_cache(models=False, datasets=True)
        assert result.deleted == 2
        assert result.freed_bytes == 250
        assert result.failures == []


class TestFolderHelperLimits:
    def test_get_folder_size_short_circuits(self, tmp_path) -> None:
        (tmp_path / "a.txt").write_text("x" * 100)
        (tmp_path / "b.txt").write_text("y" * 100)
        # limit below total -> returns as soon as exceeded
        assert hf._get_folder_size(str(tmp_path), limit=50) > 50

    def test_count_files_short_circuits(self, tmp_path) -> None:
        for i in range(5):
            (tmp_path / f"f{i}.txt").write_text("x")
        assert hf._count_files(str(tmp_path), limit=2) == 3

    def test_is_large_folder_by_file_count(self, tmp_path) -> None:
        for i in range(3):
            (tmp_path / f"f{i}.txt").write_text("x")
        assert hf._is_large_folder(str(tmp_path), file_count_threshold=2) is True


class TestGetHfApi:
    def test_returns_hf_api_instance(self) -> None:
        pytest.importorskip("huggingface_hub")
        from huggingface_hub import HfApi

        assert isinstance(hf._get_hf_api(), HfApi)


class TestColumnDetection:
    def test_audio_column_override_missing_raises(self) -> None:
        pytest.importorskip("datasets")
        with pytest.raises(InvalidInputError):
            hf._detect_audio_column({"label": object()}, "nope")

    def test_audio_column_override_present(self) -> None:
        pytest.importorskip("datasets")
        from datasets import Audio

        features = {"sound": Audio(), "label": object()}
        assert hf._detect_audio_column(features, "sound") == "sound"

    def test_label_column_override_missing_raises(self) -> None:
        pytest.importorskip("datasets")
        with pytest.raises(InvalidInputError):
            hf._detect_label_column({"a": object()}, "nope")

    def test_label_column_override_present(self) -> None:
        pytest.importorskip("datasets")
        assert hf._detect_label_column({"mylabel": object()}, "mylabel") == "mylabel"


class TestExtractAudio:
    def test_dict_form(self) -> None:
        import numpy as np

        cell = {"array": np.zeros(4), "sampling_rate": 16000, "path": "a.wav"}
        array, sr, path = hf._extract_audio(cell)
        assert sr == 16000
        assert path == "a.wav"

    def test_torchcodec_form(self) -> None:
        import numpy as np

        class _Samples:
            data = type("T", (), {"numpy": lambda self: np.ones((2, 8))})()
            sample_rate = 22050

        class _Decoder:
            def get_all_samples(self):
                return _Samples()

        array, sr, path = hf._extract_audio(_Decoder())
        assert sr == 22050
        assert path is None
        assert array.ndim == 1  # stereo averaged to mono

    def test_unrecognized_raises(self) -> None:
        with pytest.raises(InvalidInputError):
            hf._extract_audio(12345)


class TestPullDatasetFailures:
    def test_load_failure_raises_catalog_error(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("datasets")
        import datasets as hfds

        def boom(*a, **k):
            raise RuntimeError("404")

        monkeypatch.setattr(hfds, "load_dataset", boom)
        with pytest.raises(CatalogError):
            hf.pull_dataset("user/ds", str(tmp_path), split="train")

    def test_materialize_failure_raises_catalog_error(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("datasets")
        import numpy as np

        import datasets as hfds
        from datasets import Audio, Dataset, Features, Value

        ds = Dataset.from_dict(
            {
                "audio": [
                    {
                        "array": np.zeros(16000, dtype=np.float32),
                        "sampling_rate": 16000,
                        "path": "a.wav",
                    }
                ],
                "label": ["dog"],
            },
            features=Features({"audio": Audio(sampling_rate=16000), "label": Value("string")}),
        )
        monkeypatch.setattr(hfds, "load_dataset", lambda *a, **k: ds)

        def boom(*a, **k):
            raise RuntimeError("save failed")

        # save_audio is imported inside pull_dataset from bioamla.audio.
        monkeypatch.setattr("bioamla.audio.save_audio", boom)
        with pytest.raises(CatalogError):
            hf.pull_dataset("user/ds", str(tmp_path / "out"), split="train")


class _FakeRevision:
    def __init__(self, path):
        self.snapshot_path = path


class _FakeRepoWithRevisions:
    def __init__(self, repo_id, repo_type, size, revisions=None):
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.size_on_disk = size
        self.revisions = revisions or []


class TestPurgeFailure:
    def test_per_repo_failure_recorded(self, monkeypatch) -> None:
        import huggingface_hub

        repo = _FakeRepoWithRevisions("user/ds", "dataset", 100, revisions=[_FakeRevision("/x")])
        monkeypatch.setattr(huggingface_hub, "scan_cache_dir", lambda: _FakeCache([repo]))

        # Make revision iteration raise to hit the failure branch.
        def boom(path, ignore_errors=False):
            raise RuntimeError("rmtree boom")

        monkeypatch.setattr("shutil.rmtree", boom)
        result = hf.purge_cache(models=False, datasets=True)
        assert result.deleted == 0
        assert result.failures
