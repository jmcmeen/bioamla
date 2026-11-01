"""Tests for the train_ast library function (network-free units).

Full fine-tuning needs a pretrained AST checkpoint (large download), so it is
not exercised here; these cover the parameter validation and dataset-resolution
logic that run before any model is loaded.
"""

from pathlib import Path
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
        from bioamla.datasets.augmentation import WaveformAugmentationConfig
        from bioamla.ml.training import _validate_augmentation

        cfg = WaveformAugmentationConfig(add_noise=True, noise_min_snr=30.0, noise_max_snr=10.0)
        with pytest.raises(TrainingError):
            _validate_augmentation(cfg)

    def test_valid_ranges_pass(self) -> None:
        from bioamla.datasets.augmentation import WaveformAugmentationConfig
        from bioamla.ml.training import _validate_augmentation

        _validate_augmentation(WaveformAugmentationConfig())  # defaults are valid

    def test_inverted_range_ignored_when_layer_disabled(self) -> None:
        from bioamla.datasets.augmentation import WaveformAugmentationConfig
        from bioamla.ml.training import _validate_augmentation

        # add_noise is off, so its inverted SNR range is not a misconfiguration.
        _validate_augmentation(WaveformAugmentationConfig(noise_min_snr=30.0, noise_max_snr=10.0))


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
    """Training options: explicit value > config file > built-in default."""

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

    def test_no_config_path_gives_defaults_and_explicit_values(self) -> None:
        from bioamla.ml.train_config import AST_TRAIN_DEFAULTS, resolve_train_options

        out = resolve_train_options({"num_train_epochs": 3})
        assert out == {**AST_TRAIN_DEFAULTS, "num_train_epochs": 3}

    def test_config_fills_unset_options(self, tmp_path) -> None:
        from bioamla.ml.train_config import resolve_train_options

        out = resolve_train_options({}, self._write_config(tmp_path))
        assert out["base_model"] == "cfg/model"
        assert out["learning_rate"] == 1e-4
        assert out["num_train_epochs"] == 20  # epochs -> num_train_epochs
        assert out["per_device_train_batch_size"] == 32  # batch_size -> ...
        assert out["report_to"] == "mlflow"  # config overrides the tensorboard default
        assert out["eval_steps"] == 1  # not in the config -> default

    def test_explicit_option_overrides_config(self, tmp_path) -> None:
        from bioamla.ml.train_config import resolve_train_options

        cfg = self._write_config(tmp_path)
        out = resolve_train_options({"num_train_epochs": 3, "learning_rate": 9e-5}, cfg)
        assert out["num_train_epochs"] == 3  # explicit wins over config's 20
        assert out["learning_rate"] == 9e-5  # explicit wins over config's 1e-4
        assert out["per_device_train_batch_size"] == 32  # unset -> config wins

    def test_unknown_option_is_an_error(self) -> None:
        from bioamla.exceptions import InvalidInputError
        from bioamla.ml.train_config import resolve_train_options

        with pytest.raises(InvalidInputError, match="Unknown training options: epochs"):
            resolve_train_options({"epochs": 3})

    def test_cli_defaults_match_the_library(self) -> None:
        """`ast train --help` shows the library's defaults (the CLI can't import them)."""
        from bioamla.cli.commands.models import ast_train
        from bioamla.ml.train_config import AST_TRAIN_DEFAULTS

        shown = {p.name: p.default for p in ast_train.params}
        for name, default in AST_TRAIN_DEFAULTS.items():
            assert shown[name] == default, name


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

    def test_class_folders(self, tmp_path) -> None:
        from bioamla.ml.training import _load_directory_dataset

        for label, names in (("frog", ["a.wav", "nested/b.flac"]), ("toad", ["c.wav"])):
            for name in names:
                (tmp_path / label / name).parent.mkdir(parents=True, exist_ok=True)
                (tmp_path / label / name).write_bytes(b"RIFF....")
        (tmp_path / "toad" / "notes.txt").write_text("not audio")
        ds, col = _load_directory_dataset(tmp_path, str(tmp_path), "train", "category")
        assert col == "label"
        # Class names come from the folders, paths stay plain (decoded by bioamla).
        assert ds["label"] == ["frog", "frog", "toad"]
        assert [Path(p).name for p in ds["audio"]] == ["a.wav", "b.flac", "c.wav"]
        assert ds.features["audio"].dtype == "string"

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
# Progress / cancel callback
# ---------------------------------------------------------------------------


class TestProgressCallback:
    @staticmethod
    def _state(step: int) -> SimpleNamespace:
        return SimpleNamespace(
            global_step=step, max_steps=10, epoch=step / 5, is_world_process_zero=True
        )

    def test_reports_steps_loss_and_eval(self) -> None:
        from transformers import TrainerControl

        from bioamla.ml.training import _progress_callback

        seen = []
        cb = _progress_callback(seen.append, None)
        control = TrainerControl()

        cb.on_step_end(None, self._state(1), control)
        cb.on_log(None, self._state(1), control, logs={"loss": 0.9})
        cb.on_evaluate(
            None,
            self._state(5),
            control,
            metrics={"eval_accuracy": 0.5, "eval_runtime": 3.2, "epoch": 1},
        )
        cb.on_step_end(None, self._state(6), control)

        assert [p.step for p in seen] == [1, 1, 5, 6]
        assert seen[0].loss is None
        assert seen[1].loss == 0.9  # reported as soon as it is logged
        assert seen[3].loss == 0.9
        assert seen[3].total_steps == 10
        assert seen[3].epoch == 1.2
        assert seen[3].eval_metrics == {"eval_accuracy": 0.5}
        assert seen[3].to_dict()["eval_metrics"] == {"eval_accuracy": 0.5}
        assert control.should_training_stop is False

    def test_should_stop_ends_training(self) -> None:
        from transformers import TrainerControl

        from bioamla.ml.training import _progress_callback

        stop = {"now": False}
        cb = _progress_callback(None, lambda: stop["now"])
        control = TrainerControl()

        cb.on_step_end(None, self._state(1), control)
        assert not cb.cancelled
        stop["now"] = True
        # transformers' flow callback runs first and may schedule eval/save.
        control.should_evaluate = control.should_save = True
        cb.on_step_end(None, self._state(2), control)
        assert cb.cancelled
        assert control.should_training_stop is True
        assert not control.should_evaluate
        assert not control.should_save

        # ...and again at the end of the interrupted epoch.
        epoch_control = TrainerControl(should_evaluate=True, should_save=True)
        cb.on_epoch_end(None, self._state(2), epoch_control)
        assert not epoch_control.should_evaluate
        assert not epoch_control.should_save

    def test_stop_on_last_step_keeps_the_run(self) -> None:
        from transformers import TrainerControl

        from bioamla.ml.training import _progress_callback

        cb = _progress_callback(None, lambda: True)
        control = TrainerControl()
        state = SimpleNamespace(global_step=10, max_steps=10, epoch=2.0, is_world_process_zero=True)
        cb.on_step_end(None, state, control)
        assert not cb.cancelled
        assert control.should_training_stop is False


class TestTrainingHelpers:
    def test_cleanup_step_logs_instead_of_raising(self, caplog) -> None:
        from bioamla.ml.training import _cleanup_step

        def boom() -> None:
            raise RuntimeError("tracking server down")

        with caplog.at_level("WARNING", logger="bioamla"):
            _cleanup_step("end the MLflow run", boom)
        assert "Could not end the MLflow run after training: tracking server down" in caplog.text

    @pytest.mark.parametrize(
        ("report_to", "adds_mlflow", "left"),
        [
            ("tensorboard", False, ["tensorboard"]),
            (["tensorboard", "mlflow"], True, ["tensorboard"]),
            ("none", False, ["none"]),
            ("mlflow", True, ["none"]),
        ],
    )
    def test_mlflow_has_one_writer(self, report_to, adds_mlflow, left) -> None:
        from bioamla.ml.trackers import MLflowTracker
        from bioamla.ml.training import _run_trackers

        trackers, remaining = _run_trackers(None, report_to)
        assert [type(t) for t in trackers._trackers] == ([MLflowTracker] if adds_mlflow else [])
        assert remaining == left  # transformers never gets "mlflow"

    @pytest.mark.parametrize("installed", [["mlflow", "tensorboard"], ["tensorboard"]])
    def test_all_expands_to_installed_integrations(self, installed) -> None:
        from bioamla.ml.training import _run_trackers

        with patch(
            "transformers.integrations.get_available_reporting_integrations",
            return_value=installed,
        ):
            trackers, remaining = _run_trackers(None, "all")
        assert bool(trackers) is ("mlflow" in installed)  # no warning without mlflow
        assert remaining == ["tensorboard"]

    def test_callers_mlflow_tracker_is_the_only_writer(self) -> None:
        from bioamla.ml.trackers import MLflowTracker
        from bioamla.ml.training import _run_trackers

        mine = MLflowTracker(experiment="x")
        trackers, remaining = _run_trackers([mine], ["mlflow", "tensorboard"])
        assert trackers._trackers == [mine]
        assert remaining == ["tensorboard"]

    def test_failing_tracker_only_warns(self, caplog) -> None:
        from bioamla.ml.training import _Trackers

        broken, fine = MagicMock(), MagicMock()
        broken.start_run.side_effect = ConnectionError("server down")
        with caplog.at_level("WARNING", logger="bioamla"):
            _Trackers([broken, fine]).call("start_run", "run", {})
        fine.start_run.assert_called_once_with("run", {})
        assert "Could not call MagicMock.start_run: server down" in caplog.text


# ---------------------------------------------------------------------------
# Full train_ast orchestration (everything heavy mocked)
# ---------------------------------------------------------------------------


def _real_dataset_dict():
    """A tiny DatasetDict of real (silent, synthetic) WAV files, as local datasets hold
    them: paths, decoded by bioamla. Real ``Dataset``/``DatasetDict`` objects keep
    train_ast's ``isinstance`` checks happy, and nothing touches the network."""
    import tempfile

    import numpy as np
    import soundfile as sf

    from datasets import Dataset, DatasetDict

    folder = tempfile.mkdtemp(prefix="bioamla-train-")
    for c in "abcd":
        sf.write(f"{folder}/{c}.wav", np.zeros(16000, dtype=np.float32), 16000)

    def _split():
        return Dataset.from_dict(
            {"audio": [f"{folder}/{c}.wav" for c in "abcd"], "category": ["x", "y", "x", "y"]}
        )

    return DatasetDict({"train": _split(), "test": _split()})


@pytest.mark.slow
class TestTrainAst:
    def test_augmentation_validation_rejected(self) -> None:
        pytest.importorskip("audiomentations")
        from bioamla.datasets.augmentation import WaveformAugmentationConfig
        from bioamla.ml.training import train_ast

        bad = WaveformAugmentationConfig(add_noise=True, noise_min_snr=30.0, noise_max_snr=10.0)
        with pytest.raises(TrainingError):
            train_ast(train_dataset="owner/ds", augmentation=bad)

    @staticmethod
    def _mocks(fake_trainer, trainer_cls=None):
        """The heavy-dependency patches shared by the mocked full-loop tests."""
        from contextlib import ExitStack

        class _FE:
            model_input_names = ["input_values"]
            sampling_rate = 16000
            do_normalize = True
            mean = 0.0
            std = 1.0

            def __call__(self, wavs, sampling_rate=None, return_tensors=None):
                return {"input_values": torch.zeros(len(wavs), 4, 8)}

        fake_config = SimpleNamespace(num_labels=2, label2id={}, id2label={})
        stack = ExitStack()
        for target, value in [
            ("bioamla.ml.training._load_train_dataset", (_real_dataset_dict(), "category")),
            ("transformers.ASTFeatureExtractor.from_pretrained", _FE()),
            ("transformers.ASTConfig.from_pretrained", fake_config),
            ("transformers.ASTForAudioClassification.from_pretrained", MagicMock()),
            ("transformers.TrainingArguments", MagicMock()),
            ("evaluate.load", MagicMock()),
        ]:
            stack.enter_context(patch(target, return_value=value))
        stack.enter_context(
            patch("transformers.Trainer", trainer_cls or MagicMock(return_value=fake_trainer))
        )
        stack.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
        return stack

    def test_cancel_during_training_saves_nothing(self, tmp_path) -> None:
        from transformers import TrainerControl

        from bioamla.exceptions import TrainingCancelled
        from bioamla.ml.training import train_ast

        fake_trainer = MagicMock()
        fake_trainer.accelerator = None
        seen = []
        training = {"started": False}

        def make_trainer(*args, callbacks=None, **kwargs):
            def train():
                training["started"] = True
                state = SimpleNamespace(
                    global_step=1, max_steps=8, epoch=0.25, is_world_process_zero=True
                )
                callbacks[0].on_step_end(None, state, TrainerControl())

            fake_trainer.train.side_effect = train
            return fake_trainer

        with self._mocks(fake_trainer, trainer_cls=MagicMock(side_effect=make_trainer)):
            with pytest.raises(TrainingCancelled):
                train_ast(
                    train_dataset="owner/ds",
                    training_dir=str(tmp_path),
                    dataloader_num_workers=1,
                    on_progress=seen.append,
                    # Stop on the first step, not during preprocessing.
                    should_stop=lambda: training["started"],
                )

        assert [p.step for p in seen] == [1]
        fake_trainer.save_model.assert_not_called()
        assert not (tmp_path / "best_model").exists()

    def test_cancel_before_training_builds_no_trainer(self, tmp_path, caplog) -> None:
        from bioamla.exceptions import TrainingCancelled
        from bioamla.ml.training import train_ast

        trainer_cls = MagicMock()
        with self._mocks(MagicMock(), trainer_cls=trainer_cls):
            with pytest.raises(TrainingCancelled, match="before loading the dataset"):
                train_ast(
                    train_dataset="owner/ds",
                    training_dir=str(tmp_path),
                    dataloader_num_workers=1,
                    should_stop=lambda: True,
                )
        trainer_cls.assert_not_called()
        # No "started" record, so no run is left without a status record.
        assert "Training run" not in caplog.text

    def test_full_loop_mocked(self, tmp_path) -> None:
        from bioamla.ml.training import train_ast

        fake_trainer = MagicMock()
        fake_trainer.state = SimpleNamespace(
            log_history=[{"eval_accuracy": 0.75, "eval_loss": 0.3}]
        )
        fake_trainer.accelerator = None

        with self._mocks(fake_trainer):
            result = train_ast(
                train_dataset="owner/ds",
                training_dir=str(tmp_path),
                num_train_epochs=2,
                # Keep filter/map single-process: avoids forking a multi-threaded
                # test process (Python 3.13 DeprecationWarning) and runs faster.
                dataloader_num_workers=1,
            )

        assert result.epochs == 2
        assert result.final_accuracy == 0.75
        assert result.final_loss == 0.3
        fake_trainer.train.assert_called_once()
        fake_trainer.save_model.assert_called_once()

    def test_tracker_sees_a_finished_run(self, tmp_path) -> None:
        import json

        from bioamla.ml.training import train_ast

        fake_trainer = MagicMock()
        fake_trainer.state = SimpleNamespace(
            log_history=[{"eval_accuracy": 0.75, "eval_loss": 0.3}]
        )
        fake_trainer.accelerator = None

        def make_trainer(*args, callbacks=None, **kwargs):
            def train():
                state = SimpleNamespace(
                    global_step=4, max_steps=8, epoch=0.5, is_world_process_zero=True
                )
                callbacks[0].on_log(
                    None, state, None, logs={"loss": 0.5, "epoch": 0.5, "note": "x"}
                )

            fake_trainer.train.side_effect = train
            return fake_trainer

        tracker = MagicMock()
        with self._mocks(fake_trainer, trainer_cls=MagicMock(side_effect=make_trainer)):
            result = train_ast(
                train_dataset="owner/ds",
                training_dir=str(tmp_path),
                dataloader_num_workers=1,
                report_to="none",
                tracker=tracker,
            )

        name, params = tracker.start_run.call_args.args
        assert name == f"bioamla-{result.run_id}"
        assert params["run_id"] == result.run_id and params["labels"]
        tracker.log_metrics.assert_called_once_with(4, {"loss": 0.5, "epoch": 0.5})
        summary = tmp_path / "best_model" / "bioamla_run.json"
        tracker.log_artifact.assert_called_once_with(str(summary))
        assert json.loads(summary.read_text())["result"]["final_accuracy"] == 0.75
        tracker.end_run.assert_called_once_with("finished")

    @pytest.mark.parametrize("outcome", ["cancelled", "failed"])
    def test_tracker_sees_the_real_end_status(self, tmp_path, outcome) -> None:
        from transformers import TrainerControl

        from bioamla.exceptions import TrainingCancelled
        from bioamla.ml.training import train_ast

        fake_trainer = MagicMock()
        fake_trainer.accelerator = None
        training = {"started": False}

        def make_trainer(*args, callbacks=None, **kwargs):
            def train():
                if outcome == "failed":
                    raise RuntimeError("out of memory")
                training["started"] = True
                state = SimpleNamespace(
                    global_step=1, max_steps=8, epoch=0.1, is_world_process_zero=True
                )
                callbacks[0].on_step_end(None, state, TrainerControl())

            fake_trainer.train.side_effect = train
            return fake_trainer

        tracker = MagicMock()
        expected = TrainingCancelled if outcome == "cancelled" else RuntimeError
        with self._mocks(fake_trainer, trainer_cls=MagicMock(side_effect=make_trainer)):
            with pytest.raises(expected):
                train_ast(
                    train_dataset="owner/ds",
                    training_dir=str(tmp_path),
                    dataloader_num_workers=1,
                    report_to="none",
                    tracker=tracker,
                    should_stop=lambda: training["started"],
                )
        tracker.end_run.assert_called_once_with(outcome)
        tracker.log_artifact.assert_not_called()

    def test_failed_summary_write_keeps_the_run_finished(self, tmp_path, caplog) -> None:
        from pathlib import Path

        from bioamla.ml.training import train_ast

        fake_trainer = MagicMock()
        fake_trainer.state = SimpleNamespace(log_history=[])
        fake_trainer.accelerator = None
        tracker = MagicMock()
        real_write = Path.write_text

        def write_text(self, *args, **kwargs):
            if self.name == "bioamla_run.json":
                raise OSError("disk full")
            return real_write(self, *args, **kwargs)

        with self._mocks(fake_trainer), patch.object(Path, "write_text", write_text):
            result = train_ast(
                train_dataset="owner/ds",
                training_dir=str(tmp_path),
                dataloader_num_workers=1,
                report_to="none",
                tracker=tracker,
            )
        assert result.run_id
        assert "disk full" in caplog.text
        tracker.log_artifact.assert_not_called()
        tracker.end_run.assert_called_once_with("finished")


class TestDecodeExampleAudio:
    """Training audio is decoded by bioamla, not by datasets' torchcodec decoder."""

    @pytest.fixture
    def tone(self, tmp_path):
        import numpy as np
        import soundfile as sf

        path = tmp_path / "tone.wav"
        t = np.arange(8000) / 8000
        sf.write(path, np.stack([np.sin(2 * np.pi * 440 * t)] * 2, axis=1) * 0.5, 8000)
        return path

    def test_paths_bytes_and_resampling(self, tone) -> None:
        from bioamla.ml.training import decode_example_audio

        from_path = decode_example_audio(str(tone), 16000)
        assert from_path.dtype.name == "float32" and from_path.ndim == 1
        assert len(from_path) == 16000  # one second, resampled from 8 kHz
        from_dict = decode_example_audio({"path": str(tone), "bytes": None}, 16000)
        from_bytes = decode_example_audio({"path": "tone.wav", "bytes": tone.read_bytes()}, 16000)
        assert len(from_dict) == len(from_bytes) == 16000

    def test_missing_audio_is_an_error(self, tmp_path) -> None:
        from bioamla.exceptions import AudioLoadError
        from bioamla.ml.training import decode_example_audio

        with pytest.raises(AudioLoadError):
            decode_example_audio({"path": None, "bytes": None}, 16000)
        with pytest.raises(AudioLoadError):
            decode_example_audio(str(tmp_path / "missing.wav"), 16000)

    def test_training_never_loads_torchcodec(self, tmp_path) -> None:
        """A run over local files never imports torchcodec (it fails with FFmpeg 9)."""
        import subprocess
        import sys

        code = (
            "import sys, builtins\n"
            "real = builtins.__import__\n"
            "def guard(name, *a, **k):\n"
            "    if name.split('.')[0] == 'torchcodec':\n"
            "        raise ImportError('torchcodec imported')\n"
            "    return real(name, *a, **k)\n"
            "builtins.__import__ = guard\n"
            "import pandas as pd\n"
            "from bioamla.ml.training import _dataframe_to_ast_dataset, decode_example_audio\n"
            f"ds, _ = _dataframe_to_ast_dataset(pd.DataFrame({{'audio': [{str(self._wav(tmp_path))!r}], "
            "'label': ['a']}), 'label')\n"
            "print(len(decode_example_audio(ds[0]['audio'], 16000)))\n"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "16000"

    @staticmethod
    def _wav(folder):
        import numpy as np
        import soundfile as sf

        path = folder / "a.wav"
        sf.write(path, np.zeros(16000, dtype=np.float32), 16000)
        return path


class _LengthFeatures:
    """A picklable stand-in feature extractor: each wave's length."""

    sampling_rate = 16000
    model_input_names = ["input_values"]

    def __call__(self, wavs, sampling_rate=None, return_tensors=None):
        return {"input_values": [len(w) for w in wavs]}


def test_the_batch_preprocessor_pickles_for_dataloader_workers(tmp_path) -> None:
    """Spawned DataLoader workers (macOS, Windows) pickle the dataset's transform."""
    import pickle

    import numpy as np
    import soundfile as sf

    from bioamla.ml.training import BatchPreprocessor

    sf.write(tmp_path / "a.wav", np.zeros(8000, dtype=np.float32), 8000)
    pre = pickle.loads(pickle.dumps(BatchPreprocessor(_LengthFeatures())))
    out = pre({"input_values": [str(tmp_path / "a.wav")], "labels": [1]})
    assert out == {"input_values": [16000], "labels": [1]}
