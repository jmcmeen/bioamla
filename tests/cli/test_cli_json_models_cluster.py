"""`bioamla --json` results for the models and cluster commands."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli


def _json(*args: str) -> dict:
    result = CliRunner().invoke(cli, ["--json", *args])
    assert result.exit_code == 0, result.output
    envelope = json.loads(result.stdout)
    assert envelope["ok"] is True
    return envelope["result"]


def _pred(label, conf, start, end):
    return SimpleNamespace(
        filepath="x.wav", start_time=start, end_time=end, predicted_label=label, confidence=conf
    )


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "x.wav"
    path.write_bytes(b"")
    return str(path)


def test_predict_segments(wav) -> None:
    inst = MagicMock()
    inst.predict_segments.return_value = [_pred("frog", 0.9, 0, 3), _pred("bird", 0.2, 3, 6)]
    with patch("bioamla.ml.inference.ASTInference", return_value=inst):
        result = _json(
            "models", "ast", "predict", wav, "--segment-seconds", "3", "--min-confidence", "0.5"
        )
    assert result["count"] == 1
    assert result["predictions"] == [
        {"start_time": 0, "end_time": 3, "label": "frog", "confidence": 0.9}
    ]
    assert result["params"]["segment_seconds"] == 3 and result["output"] is None


def test_predict_with_output_reports_path(wav, tmp_path) -> None:
    inst = MagicMock()
    inst.predict.return_value = _pred("frog", 0.8, 0, 10)
    out = tmp_path / "p.csv"
    with patch("bioamla.ml.inference.ASTInference", return_value=inst):
        result = _json("models", "ast", "predict", wav, "-o", str(out))
    assert result["output"] == str(out) and out.exists()


def test_evaluate(tmp_path) -> None:
    pytest.importorskip("torch")
    from bioamla.ml.ast_service import EvaluationResult

    metrics = EvaluationResult(
        accuracy=0.9, precision=0.8, recall=0.7, f1_score=0.75, total_samples=10
    )
    gt = tmp_path / "gt.csv"
    gt.write_text("file_name,label\n")
    with patch("bioamla.ml.ast_service.evaluate_directory", return_value=metrics):
        result = _json("models", "ast", "evaluate", str(tmp_path), "-g", str(gt))
    assert result["metrics"]["accuracy"] == 0.9 and result["metrics"]["total_samples"] == 10
    assert "quiet" not in result["params"]


def test_embed_reports_shape_not_vectors(wav, tmp_path) -> None:
    out = tmp_path / "e.npy"
    with patch(
        "bioamla.ml.ast_service.extract_embeddings_file",
        return_value={"embeddings": np.zeros((4, 768), dtype=np.float32)},
    ):
        result = _json("models", "ast", "embed", wav, "--model-path", "m", "-o", str(out))
    assert result["shape"] == [4, 768] and result["dtype"] == "float32"
    assert result["output"] == str(out) and out.exists()


def test_info_reports_every_label() -> None:
    labels = [f"species_{i}" for i in range(25)]
    info = {
        "path": "m",
        "model_type": "audio-spectrogram-transformer",
        "num_classes": 25,
        "classes": labels[:10],
        "has_more_classes": True,
        "labels": labels,
        "hidden_size": 768,
    }
    with patch("bioamla.ml.ast_service.get_model_info", return_value=info):
        result = _json("models", "ast", "info", "m")
    assert result["labels"] == labels  # not cut to 10 like the human view


@pytest.fixture
def embeddings(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "emb.npy"
    np.save(path, np.vstack([rng.normal(0, 1, (20, 6)), rng.normal(8, 1, (20, 6))]))
    return path


def test_cluster_commands(embeddings, tmp_path) -> None:
    pytest.importorskip("sklearn")
    reduced = _json("cluster", "reduce", str(embeddings), "-o", str(tmp_path / "r.npy"))
    assert reduced["shape"] == [40, 2]

    labels = tmp_path / "l.npy"
    fit = _json("cluster", "fit", str(embeddings), "-o", str(labels), "-m", "kmeans", "-k", "2")
    assert fit["n_clusters"] == 2 and sum(fit["cluster_sizes"].values()) == 40

    analysis = _json("cluster", "analyze", str(embeddings), str(labels))
    assert analysis["n_samples"] == 40 and analysis["labels_file"] == str(labels)
    json.dumps(analysis)  # cluster_stats (numpy inside) is JSON-clean

    novelty = _json("cluster", "novelty", str(embeddings), "-o", str(tmp_path / "n.npy"))
    assert novelty["n_novel"] == len(novelty["novel_indices"])


@pytest.mark.parametrize("content", ["missing", "empty", "one_d"])
def test_bad_embeddings_are_invalid_input(tmp_path, content) -> None:
    path = tmp_path / "e.npy"
    if content == "empty":
        path.write_bytes(b"")
    elif content == "one_d":
        np.save(path, np.zeros(5))
    result = CliRunner().invoke(cli, ["--json", "cluster", "reduce", str(path), "-o", "x.npy"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"]["type"] == "InvalidInputError"


def test_model_labels_follow_class_index_order() -> None:
    pytest.importorskip("transformers")
    from bioamla.ml.embedding import get_ast_model_info

    # Keys as HF often stores them: sorted as strings.
    id2label = {i: f"label_{i}" for i in sorted(range(12), key=str)}
    config = SimpleNamespace(id2label=id2label, model_type="ast", hidden_size=768)
    with patch("transformers.AutoConfig.from_pretrained", return_value=config):
        info = get_ast_model_info("m")
    assert info["labels"] == [f"label_{i}" for i in range(12)]
    assert info["classes"] == [f"label_{i}" for i in range(10)]


def test_evaluate_csv_report_is_csv(tmp_path) -> None:
    pytest.importorskip("torch")
    import csv

    from bioamla.ml.ast_service import EvaluationResult

    metrics = EvaluationResult(
        accuracy=0.9, precision=0.8, recall=0.7, f1_score=0.75, total_samples=10
    )
    gt = tmp_path / "gt.csv"
    gt.write_text("file_name,label\n")
    out = tmp_path / "eval.csv"
    with patch("bioamla.ml.ast_service.evaluate_directory", return_value=metrics):
        result = CliRunner().invoke(
            cli,
            [
                "models",
                "ast",
                "evaluate",
                str(tmp_path),
                "-g",
                str(gt),
                "--format",
                "csv",
                "-o",
                str(out),
            ],
        )
    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader(out.open()))
    # Long format: the overall row, then one row per class (none in this result).
    assert rows == [
        {
            "label": "(overall)",
            "accuracy": "0.9",
            "precision": "0.8",
            "recall": "0.7",
            "f1_score": "0.75",
            "support": "10",
        }
    ]
