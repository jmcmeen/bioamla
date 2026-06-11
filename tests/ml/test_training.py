"""Tests for the train_ast library function (network-free units).

Full fine-tuning needs a pretrained AST checkpoint (large download), so it is
not exercised here; these cover the parameter validation and dataset-resolution
logic that run before any model is loaded.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from bioamla.exceptions import TrainingError

torch = pytest.importorskip("torch")
pytest.importorskip("datasets")
pytest.importorskip("transformers")


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

    def test_resolves_paths_and_columns(self, tmp_path) -> None:
        import pandas as pd

        from bioamla.ml.training import _load_csv_dataset

        csv = tmp_path / "meta.csv"
        pd.DataFrame({"filename": ["a.wav", "b.wav"], "species": ["x", "y"]}).to_csv(
            csv, index=False
        )
        with patch(
            "bioamla.ml.training._dataframe_to_ast_dataset", return_value=("DS", False)
        ) as conv:
            ds, label_col = _load_csv_dataset(csv, "category")
        assert ds == "DS"
        assert label_col == "species"
        # absolute audio path was built relative to the csv dir
        passed_df = conv.call_args[0][0]
        assert passed_df["audio"].iloc[0] == str(tmp_path / "a.wav")

    def test_fixed_split_logs(self, tmp_path) -> None:
        import pandas as pd

        from bioamla.ml.training import _load_csv_dataset

        csv = tmp_path / "meta.csv"
        pd.DataFrame({"file_name": ["a.wav"], "label": ["x"]}).to_csv(csv, index=False)
        fake_dd = {"train": [1], "test": [2]}
        with patch("bioamla.ml.training._dataframe_to_ast_dataset", return_value=(fake_dd, True)):
            ds, label_col = _load_csv_dataset(csv, "category")
        assert ds is fake_dd


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

    def test_audiofolder(self, tmp_path) -> None:
        from bioamla.ml.training import _load_directory_dataset

        cls = tmp_path / "frog"
        cls.mkdir()
        (cls / "a.wav").write_bytes(b"RIFF....")
        with patch("datasets.load_dataset", return_value="DS") as ld:
            ds, col = _load_directory_dataset(tmp_path, str(tmp_path), "train", "category")
        assert ds == "DS"
        assert col == "label"
        ld.assert_called_once_with("audiofolder", data_dir=str(tmp_path), split="train")

    def test_csv_in_directory_hint(self, tmp_path) -> None:
        from bioamla.ml.training import _load_directory_dataset

        (tmp_path / "data.csv").write_text("x")
        with pytest.raises(TrainingError, match="specify the CSV directly"):
            _load_directory_dataset(tmp_path, str(tmp_path), "train", "category")


# ---------------------------------------------------------------------------
# Dataset helpers (real, tiny HF datasets)
# ---------------------------------------------------------------------------


class TestDataframeToAstDataset:
    def test_no_split_column_returns_flat(self) -> None:
        import pandas as pd

        from bioamla.ml.training import _dataframe_to_ast_dataset
        from datasets import Dataset

        df = pd.DataFrame({"audio": ["a.wav", "b.wav"], "label": ["x", "y"]})
        ds, used = _dataframe_to_ast_dataset(df, "label")
        assert used is False
        assert isinstance(ds, Dataset)

    def test_unrecognized_split_falls_back_to_flat(self) -> None:
        import pandas as pd

        from bioamla.ml.training import _dataframe_to_ast_dataset
        from datasets import Dataset

        df = pd.DataFrame({"audio": ["a.wav"], "label": ["x"], "split": ["weird-unknown"]})
        ds, used = _dataframe_to_ast_dataset(df, "label")
        assert used is False
        assert isinstance(ds, Dataset)

    def test_fixed_split_returns_dict(self) -> None:
        import pandas as pd

        from bioamla.ml.training import _dataframe_to_ast_dataset
        from datasets import DatasetDict

        df = pd.DataFrame(
            {
                "audio": ["a.wav", "b.wav"],
                "label": ["x", "y"],
                "split": ["train", "test"],
            }
        )
        ds, used = _dataframe_to_ast_dataset(df, "label")
        assert used is True
        assert isinstance(ds, DatasetDict)
        assert set(ds.keys()) == {"train", "test"}

    def test_validation_promoted_to_test(self) -> None:
        import pandas as pd

        from bioamla.ml.training import _dataframe_to_ast_dataset

        df = pd.DataFrame(
            {
                "audio": ["a.wav", "b.wav"],
                "label": ["x", "y"],
                "split": ["train", "val"],
            }
        )
        ds, used = _dataframe_to_ast_dataset(df, "label")
        assert used is True
        assert "test" in ds.keys()  # validation -> test (no test present)


class TestLoadTrainDatasetRouting:
    def test_existing_file_non_csv_raises(self, tmp_path) -> None:
        from bioamla.ml.training import _load_train_dataset

        f = tmp_path / "data.txt"
        f.write_text("x")
        with pytest.raises(TrainingError, match="neither a CSV file nor a directory"):
            _load_train_dataset(str(f), "train", "category")

    def test_hub_with_config(self) -> None:
        from bioamla.ml.training import _load_train_dataset

        with patch("datasets.load_dataset", return_value="DS") as ld:
            out, col = _load_train_dataset("owner/name:HSN", "train", "category")
        assert out == "DS"
        ld.assert_called_once_with("owner/name", "HSN", split="train")

    def test_birdset_default(self) -> None:
        from bioamla.ml.training import _load_train_dataset

        with patch("datasets.load_dataset", return_value="DS") as ld:
            _load_train_dataset("owner/BirdSet", "train", "category")
        ld.assert_called_once_with("owner/BirdSet", "HSN", split="train")

    def test_plain_hub_id(self) -> None:
        from bioamla.ml.training import _load_train_dataset

        with patch("datasets.load_dataset", return_value="DS") as ld:
            _load_train_dataset("owner/dataset", "train", "category")
        ld.assert_called_once_with("owner/dataset", split="train")

    def test_dir_with_metadata_routes_to_csv(self, tmp_path) -> None:
        from bioamla.ml.training import _load_train_dataset

        (tmp_path / "metadata.csv").write_text("file_name,label\na.wav,x\n")
        with patch("bioamla.ml.training._load_csv_dataset", return_value=("DS", "label")) as lc:
            out, col = _load_train_dataset(str(tmp_path), "train", "category")
        assert out == "DS"
        lc.assert_called_once()


# ---------------------------------------------------------------------------
# Full train_ast orchestration (everything heavy mocked)
# ---------------------------------------------------------------------------


def _real_dataset_dict():
    """A tiny in-memory DatasetDict with real (synthetic) audio arrays.

    Using real ``Dataset``/``DatasetDict`` objects keeps train_ast's many
    ``isinstance`` checks happy; the audio is silent synthetic arrays so the
    map/filter/cast/transform machinery runs without touching disk or network.
    """
    import numpy as np

    from datasets import Audio, Dataset, DatasetDict

    def _split():
        rows = {
            "audio": [
                {
                    "array": np.zeros(16000, dtype=np.float32),
                    "sampling_rate": 16000,
                    "path": f"{c}.wav",
                }
                for c in "abcd"
            ],
            "category": ["x", "y", "x", "y"],
        }
        ds = Dataset.from_dict(rows)
        return ds.cast_column("audio", Audio(sampling_rate=16000))

    return DatasetDict({"train": _split(), "test": _split()})


@pytest.mark.slow
class TestTrainAst:
    def test_augmentation_validation_rejected(self) -> None:
        pytest.importorskip("audiomentations")
        from bioamla.datasets.augmentation import AugmentationConfig
        from bioamla.ml.training import train_ast

        bad = AugmentationConfig(noise_min_snr=30.0, noise_max_snr=10.0)
        with pytest.raises(TrainingError):
            train_ast(train_dataset="owner/ds", augmentation=bad)

    def test_full_loop_mocked(self, tmp_path) -> None:
        from bioamla.ml.training import train_ast

        dd = _real_dataset_dict()

        # A feature extractor stub: returns a real tensor batch so compute_stats
        # and the preprocess transforms run for real (just no learned weights).
        class _FE:
            model_input_names = ["input_values"]
            sampling_rate = 16000
            do_normalize = True
            mean = 0.0
            std = 1.0

            def __call__(self, wavs, sampling_rate=None, return_tensors=None):
                n = len(wavs)
                return {"input_values": torch.zeros(n, 4, 8)}

        fe = _FE()

        fake_trainer = MagicMock()
        fake_trainer.state = SimpleNamespace(
            log_history=[{"eval_accuracy": 0.75, "eval_loss": 0.3}]
        )
        fake_trainer.accelerator = None

        fake_config = SimpleNamespace(num_labels=2, label2id={}, id2label={})

        with (
            patch("bioamla.ml.training._load_train_dataset", return_value=(dd, "category")),
            patch("transformers.ASTFeatureExtractor.from_pretrained", return_value=fe),
            patch("transformers.ASTConfig.from_pretrained", return_value=fake_config),
            patch(
                "transformers.ASTForAudioClassification.from_pretrained",
                return_value=MagicMock(),
            ),
            patch("transformers.Trainer", return_value=fake_trainer),
            patch("transformers.TrainingArguments", return_value=MagicMock()),
            patch("evaluate.load", return_value=MagicMock()),
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            result = train_ast(
                train_dataset="owner/ds",
                training_dir=str(tmp_path),
                num_train_epochs=2,
            )

        assert result.epochs == 2
        assert result.final_accuracy == 0.75
        assert result.final_loss == 0.3
        fake_trainer.train.assert_called_once()
        fake_trainer.save_model.assert_called_once()
