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


class TestLoadPartitionedDirectory:
    """A dataset dir with a metadata.csv (post-partition) loads via the CSV."""

    def test_directory_with_metadata_honors_split(self, tmp_path) -> None:
        pytest.importorskip("datasets")
        import csv

        import numpy as np
        import soundfile as sf

        from bioamla.ml.training import _load_train_dataset
        from datasets import DatasetDict

        # Mirror `dataset partition --mode subdirs`: data/<split>/<label>/*.wav + metadata.csv.
        root = tmp_path / "data"
        rows = []
        for split in ("train", "test"):
            for label in ("frog", "bird"):
                d = root / split / label
                d.mkdir(parents=True)
                rel = f"{split}/{label}/clip.wav"
                sf.write(str(root / rel), np.zeros(16000, dtype="float32"), 16000)
                rows.append({"file_name": rel, "label": label, "split": split})
        with open(root / "metadata.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file_name", "label", "split"])
            w.writeheader()
            w.writerows(rows)

        dataset, label_col = _load_train_dataset(str(root), "train", "category")
        assert label_col == "label"
        assert isinstance(dataset, DatasetDict)
        assert set(dataset.keys()) >= {"train", "test"}
        assert len(dataset["train"]) == 2 and len(dataset["test"]) == 2


class TestASTCheckpointLoading:
    """Guard that the pretrained AST encoder actually loads (transformers >= 5.10.2).

    transformers 5.x renamed AST's modules (encoder.layer.N.attention.attention.query
    -> layers.N.attention.q_proj) and ships a key-conversion for the old Hub
    checkpoints. If a future bump drops that conversion, fine-tuning would silently
    train a random encoder — this catches it without downloading the real checkpoint.
    """

    def test_old_format_checkpoint_loads_clean(self, tmp_path) -> None:
        torch = pytest.importorskip("torch")
        pytest.importorskip("transformers")
        import safetensors.torch as st
        from transformers import ASTConfig, ASTForAudioClassification

        cfg = ASTConfig(
            num_hidden_layers=2,
            hidden_size=32,
            num_attention_heads=2,
            intermediate_size=64,
            num_mel_bins=16,
            max_length=32,
            patch_size=16,
            frequency_stride=16,
            time_stride=16,
            num_labels=5,
        )
        torch.manual_seed(0)
        model = ASTForAudioClassification(cfg)
        new_sd = model.state_dict()

        # Rewrite to the legacy MIT checkpoint key layout.
        def new_to_old(k: str) -> str:
            k = k.replace(
                "audio_spectrogram_transformer.layers.",
                "audio_spectrogram_transformer.encoder.layer.",
            )
            k = k.replace(".attention.q_proj.", ".attention.attention.query.")
            k = k.replace(".attention.k_proj.", ".attention.attention.key.")
            k = k.replace(".attention.v_proj.", ".attention.attention.value.")
            k = k.replace(".attention.o_proj.", ".attention.output.dense.")
            k = k.replace(".mlp.fc1.", ".intermediate.dense.")
            k = k.replace(".mlp.fc2.", ".output.dense.")
            return k

        ckpt = tmp_path / "legacy"
        ckpt.mkdir()
        st.save_file(
            {new_to_old(k): v for k, v in new_sd.items()},
            str(ckpt / "model.safetensors"),
            metadata={"format": "pt"},
        )
        cfg.save_pretrained(ckpt)

        loaded, info = ASTForAudioClassification.from_pretrained(ckpt, output_loading_info=True)
        assert len(info["missing_keys"]) == 0
        assert len(info["unexpected_keys"]) == 0
        assert torch.allclose(
            loaded.audio_spectrogram_transformer.layers[0].attention.q_proj.weight,
            new_sd["audio_spectrogram_transformer.layers.0.attention.q_proj.weight"],
        )


class TestTrainConfigOverlay:
    """`--config` precedence: CLI flag > config file > built-in default."""

    def _ctx(self, commandline_params):
        from click.core import ParameterSource

        class FakeCtx:
            def get_parameter_source(self, name):
                if name in commandline_params:
                    return ParameterSource.COMMANDLINE
                return ParameterSource.DEFAULT

        return FakeCtx()

    def _write_config(self, tmp_path):
        cfg = tmp_path / "train.toml"
        cfg.write_text(
            "[models]\n"
            'default_ast_model = "cfg/model"\n\n'
            "[training]\n"
            "learning_rate = 1e-4\n"
            "epochs = 20\n"
            "batch_size = 32\n"
            'report_to = "mlflow"\n'
        )
        return str(cfg)

    def test_no_config_path_is_passthrough(self) -> None:
        from bioamla.cli.commands.models import _apply_train_config

        values = {"learning_rate": 5e-5, "num_train_epochs": 1}
        assert _apply_train_config(self._ctx(set()), None, dict(values)) == values

    def test_config_fills_defaulted_flags(self, tmp_path) -> None:
        from bioamla.cli.commands.models import _apply_train_config

        cfg = self._write_config(tmp_path)
        values = {
            "base_model": "MIT/ast-finetuned-audioset-10-10-0.4593",
            "learning_rate": 5e-5,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 8,
            "eval_steps": 1,
            "report_to": "tensorboard",
        }
        out = _apply_train_config(self._ctx(set()), cfg, dict(values))
        assert out["base_model"] == "cfg/model"
        assert out["learning_rate"] == 1e-4
        assert out["num_train_epochs"] == 20  # epochs -> num_train_epochs
        assert out["per_device_train_batch_size"] == 32  # batch_size -> ...
        assert out["report_to"] == "mlflow"  # config overrides the tensorboard default
        # Not present in config -> keeps the flag/default value.
        assert out["eval_steps"] == 1

    def test_explicit_flag_overrides_config(self, tmp_path) -> None:
        from bioamla.cli.commands.models import _apply_train_config

        cfg = self._write_config(tmp_path)
        # User explicitly passed --num-train-epochs and --learning-rate.
        ctx = self._ctx({"num_train_epochs", "learning_rate"})
        values = {"num_train_epochs": 3, "learning_rate": 9e-5, "per_device_train_batch_size": 8}
        out = _apply_train_config(ctx, cfg, dict(values))
        assert out["num_train_epochs"] == 3  # flag wins over config's 20
        assert out["learning_rate"] == 9e-5  # flag wins over config's 1e-4
        assert out["per_device_train_batch_size"] == 32  # defaulted -> config wins


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
