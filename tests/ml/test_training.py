"""Tests for the train_ast library function (network-free units).

Full fine-tuning needs a pretrained AST checkpoint (large download), so it is
not exercised here; these cover the parameter validation and dataset-resolution
logic that run before any model is loaded.
"""

import pytest

from bioamla.exceptions import TrainingError


class TestValidateAugmentation:
    def test_inverted_range_raises(self) -> None:
        pytest.importorskip("audiomentations")
        from bioamla.datasets.augmentation import AugmentationConfig
        from bioamla.ml.training import _validate_augmentation

        cfg = AugmentationConfig(noise_min_snr=30.0, noise_max_snr=10.0)
        with pytest.raises(TrainingError):
            _validate_augmentation(cfg)

    def test_valid_ranges_pass(self) -> None:
        from bioamla.datasets.augmentation import AugmentationConfig
        from bioamla.ml.training import _validate_augmentation

        _validate_augmentation(AugmentationConfig())  # defaults are valid


class TestLoadCsvDataset:
    def test_missing_file_column_raises(self, tmp_path) -> None:
        pytest.importorskip("datasets")
        import pandas as pd

        from bioamla.ml.training import _load_csv_dataset

        csv = tmp_path / "meta.csv"
        pd.DataFrame({"label": ["a"], "notafile": ["x.wav"]}).to_csv(csv, index=False)
        with pytest.raises(TrainingError, match="file column"):
            _load_csv_dataset(csv, "category")

    def test_missing_label_column_raises(self, tmp_path) -> None:
        pytest.importorskip("datasets")
        import pandas as pd

        from bioamla.ml.training import _load_csv_dataset

        csv = tmp_path / "meta.csv"
        pd.DataFrame({"file_name": ["x.wav"], "notalabel": ["a"]}).to_csv(csv, index=False)
        with pytest.raises(TrainingError, match="label column"):
            _load_csv_dataset(csv, "category")


class TestLoadDirectoryDataset:
    def test_subdirs_without_audio_raises(self, tmp_path) -> None:
        pytest.importorskip("datasets")
        from bioamla.ml.training import _load_directory_dataset

        (tmp_path / "classA").mkdir()
        (tmp_path / "classA" / "notes.txt").write_text("x")
        with pytest.raises(TrainingError):
            _load_directory_dataset(tmp_path, str(tmp_path), "train", "category")

    def test_empty_directory_raises(self, tmp_path) -> None:
        pytest.importorskip("datasets")
        from bioamla.ml.training import _load_directory_dataset

        with pytest.raises(TrainingError):
            _load_directory_dataset(tmp_path, str(tmp_path), "train", "category")
