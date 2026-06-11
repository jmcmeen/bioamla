"""CLI tests for `bioamla models ast` commands."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli
from bioamla.exceptions import ModelError, TrainingError


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _pred(label="frog", conf=0.9, start=0.0, end=3.0):
    return SimpleNamespace(
        filepath="x.wav",
        start_time=start,
        end_time=end,
        predicted_label=label,
        confidence=conf,
    )


def test_models_group_help(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["models", "--help"])
    assert result.exit_code == 0
    assert "ast" in result.output


def test_ast_group_help(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["models", "ast", "--help"])
    assert result.exit_code == 0
    for sub in ["predict", "train", "evaluate", "embed", "info"]:
        assert sub in result.output


# --- predict -------------------------------------------------------------


def test_ast_predict_whole_file(runner: CliRunner, test_audio_path) -> None:
    inst = MagicMock()
    inst.predict.return_value = _pred()
    with patch("bioamla.ml.ASTInference", return_value=inst):
        result = runner.invoke(cli, ["models", "ast", "predict", test_audio_path])
    assert result.exit_code == 0, result.output
    assert "frog" in result.output


def test_ast_predict_segments(runner: CliRunner, test_audio_path) -> None:
    inst = MagicMock()
    inst.predict_segments.return_value = [_pred(start=0, end=3), _pred(start=3, end=6)]
    with patch("bioamla.ml.ASTInference", return_value=inst):
        result = runner.invoke(
            cli, ["models", "ast", "predict", test_audio_path, "--segment-duration", "3"]
        )
    assert result.exit_code == 0, result.output
    assert "0.00-3.00s" in result.output


def test_ast_predict_output_csv(runner: CliRunner, test_audio_path, tmp_path) -> None:
    out = tmp_path / "preds.csv"
    inst = MagicMock()
    inst.predict.return_value = _pred()
    with patch("bioamla.ml.ASTInference", return_value=inst):
        result = runner.invoke(cli, ["models", "ast", "predict", test_audio_path, "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "prediction" in out.read_text()


def test_ast_predict_min_confidence(runner: CliRunner, test_audio_path) -> None:
    inst = MagicMock()
    inst.predict.return_value = _pred(conf=0.1)
    with patch("bioamla.ml.ASTInference", return_value=inst):
        result = runner.invoke(
            cli, ["models", "ast", "predict", test_audio_path, "--min-confidence", "0.5"]
        )
    assert result.exit_code == 0
    assert "frog" not in result.output


def test_ast_predict_error(runner: CliRunner, test_audio_path) -> None:
    with patch("bioamla.ml.ASTInference", side_effect=ModelError("load failed")):
        result = runner.invoke(cli, ["models", "ast", "predict", test_audio_path])
    assert result.exit_code != 0


def test_ast_predict_missing_file(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["models", "ast", "predict", "/no/such/file.wav"])
    assert result.exit_code != 0


# --- train ---------------------------------------------------------------


def test_ast_train(runner: CliRunner, tmp_path) -> None:
    result_obj = SimpleNamespace(
        model_path=str(tmp_path / "model"), final_accuracy=0.92, final_loss=0.3
    )
    with (
        patch("bioamla.ml.train_ast", return_value=result_obj) as m,
        patch("bioamla.cli.logging_setup.configure_cli_logging"),
    ):
        result = runner.invoke(
            cli,
            ["models", "ast", "train", "--train-dataset", "me/ds", "--training-dir", str(tmp_path)],
        )
    assert result.exit_code == 0, result.output
    assert "Training complete" in result.output
    m.assert_called_once()


def test_ast_train_no_augment(runner: CliRunner, tmp_path) -> None:
    result_obj = SimpleNamespace(model_path="m", final_accuracy=None, final_loss=None)
    with (
        patch("bioamla.ml.train_ast", return_value=result_obj) as m,
        patch("bioamla.cli.logging_setup.configure_cli_logging"),
    ):
        result = runner.invoke(
            cli, ["models", "ast", "train", "--train-dataset", "me/ds", "--no-augment"]
        )
    assert result.exit_code == 0, result.output
    # augmentation should be None when --no-augment
    assert m.call_args.kwargs["augmentation"] is None


def test_ast_train_with_config(runner: CliRunner, tmp_path) -> None:
    cfg = tmp_path / "bioamla.toml"
    cfg.write_text(
        "[training]\nepochs = 7\nlearning_rate = 0.001\n[models]\ndefault_ast_model = 'foo/bar'\n"
    )
    result_obj = SimpleNamespace(model_path="m", final_accuracy=None, final_loss=None)
    with (
        patch("bioamla.ml.train_ast", return_value=result_obj) as m,
        patch("bioamla.cli.logging_setup.configure_cli_logging"),
    ):
        result = runner.invoke(
            cli,
            ["models", "ast", "train", "--train-dataset", "me/ds", "--config", str(cfg)],
        )
    assert result.exit_code == 0, result.output
    # config values overlay defaults
    assert m.call_args.kwargs["num_train_epochs"] == 7
    assert m.call_args.kwargs["base_model"] == "foo/bar"


def test_ast_train_error(runner: CliRunner) -> None:
    with (
        patch("bioamla.ml.train_ast", side_effect=TrainingError("empty dataset")),
        patch("bioamla.cli.logging_setup.configure_cli_logging"),
    ):
        result = runner.invoke(cli, ["models", "ast", "train", "--train-dataset", "me/ds"])
    assert result.exit_code != 0


# --- evaluate ------------------------------------------------------------


def test_ast_evaluate(runner: CliRunner, test_audio_dir, tmp_path) -> None:
    gt = tmp_path / "gt.csv"
    gt.write_text("file_name,label\na.wav,frog\n")
    eval_result = SimpleNamespace(
        accuracy=0.9,
        precision=0.88,
        recall=0.91,
        f1_score=0.89,
        total_samples=10,
        to_dict=lambda: {"accuracy": 0.9},
    )
    with patch("bioamla.ml.evaluate_directory", return_value=eval_result):
        result = runner.invoke(cli, ["models", "ast", "evaluate", test_audio_dir, "-g", str(gt)])
    assert result.exit_code == 0, result.output
    assert "Accuracy: 0.9000" in result.output


def test_ast_evaluate_output_json(runner: CliRunner, test_audio_dir, tmp_path) -> None:
    gt = tmp_path / "gt.csv"
    gt.write_text("file_name,label\na.wav,frog\n")
    out = tmp_path / "res.json"
    eval_result = SimpleNamespace(
        accuracy=0.9,
        precision=0.8,
        recall=0.8,
        f1_score=0.8,
        total_samples=5,
        to_dict=lambda: {"accuracy": 0.9},
    )
    with patch("bioamla.ml.evaluate_directory", return_value=eval_result):
        result = runner.invoke(
            cli,
            [
                "models",
                "ast",
                "evaluate",
                test_audio_dir,
                "-g",
                str(gt),
                "-o",
                str(out),
                "--format",
                "json",
            ],
        )
    assert result.exit_code == 0
    assert out.exists()
    assert "accuracy" in out.read_text()


def test_ast_evaluate_error(runner: CliRunner, test_audio_dir, tmp_path) -> None:
    gt = tmp_path / "gt.csv"
    gt.write_text("file_name,label\na.wav,frog\n")
    with patch("bioamla.ml.evaluate_directory", side_effect=ModelError("bad model")):
        result = runner.invoke(cli, ["models", "ast", "evaluate", test_audio_dir, "-g", str(gt)])
    assert result.exit_code != 0


# --- embed ---------------------------------------------------------------


def test_ast_embed(runner: CliRunner, test_audio_path, tmp_path) -> None:
    out = tmp_path / "emb.npy"
    with patch(
        "bioamla.ml.extract_embeddings_file",
        return_value={"embeddings": np.zeros((1, 768), dtype=np.float32)},
    ):
        result = runner.invoke(
            cli,
            ["models", "ast", "embed", test_audio_path, "--model-path", "m", "-o", str(out)],
        )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_ast_embed_error(runner: CliRunner, test_audio_path, tmp_path) -> None:
    out = tmp_path / "emb.npy"
    with patch("bioamla.ml.extract_embeddings_file", side_effect=ModelError("x")):
        result = runner.invoke(
            cli,
            ["models", "ast", "embed", test_audio_path, "--model-path", "m", "-o", str(out)],
        )
    assert result.exit_code != 0


# --- info ----------------------------------------------------------------


def test_ast_info(runner: CliRunner) -> None:
    info = {
        "path": "me/model",
        "model_type": "ast",
        "num_classes": 3,
        "classes": ["a", "b", "c"],
        "has_more_classes": False,
    }
    with patch("bioamla.ml.get_model_info", return_value=info):
        result = runner.invoke(cli, ["models", "ast", "info", "me/model"])
    assert result.exit_code == 0, result.output
    assert "me/model" in result.output
    assert "Classes: 3" in result.output


def test_ast_info_more_classes(runner: CliRunner) -> None:
    info = {
        "path": "me/model",
        "model_type": "ast",
        "num_classes": 15,
        "classes": [str(i) for i in range(10)],
        "has_more_classes": True,
    }
    with patch("bioamla.ml.get_model_info", return_value=info):
        result = runner.invoke(cli, ["models", "ast", "info", "me/model"])
    assert result.exit_code == 0
    assert "more" in result.output


def test_ast_info_error(runner: CliRunner) -> None:
    with patch("bioamla.ml.get_model_info", side_effect=ModelError("not found")):
        result = runner.invoke(cli, ["models", "ast", "info", "me/model"])
    assert result.exit_code != 0
