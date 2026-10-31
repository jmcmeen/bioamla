"""evaluate_directory: per-class metrics, batched inference, skipped files."""

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from bioamla.ml.ast_service import _compute_metrics, evaluate_directory  # noqa: E402


def test_per_class_metrics_and_confusion_matrix() -> None:
    y_true = ["frog", "frog", "frog", "bird", "bird"]
    y_pred = ["frog", "frog", "bird", "bird", "frog"]
    m = _compute_metrics(y_true, y_pred)

    assert m["labels"] == ["bird", "frog"]
    # rows = true, cols = predicted, in labels order
    assert m["confusion_matrix"] == [[1, 1], [1, 2]]
    frog = m["per_class"]["frog"]
    assert frog["support"] == 3
    assert frog["precision"] == pytest.approx(2 / 3) and frog["recall"] == pytest.approx(2 / 3)
    bird = m["per_class"]["bird"]
    assert (bird["precision"], bird["recall"], bird["support"]) == (0.5, 0.5, 2)
    assert m["f1_score"] == pytest.approx((frog["f1_score"] + bird["f1_score"]) / 2)


def _setup(tmp_path, names):
    for name in names:
        (tmp_path / name).write_bytes(b"")
    gt = tmp_path / "gt.csv"
    gt.write_text(
        "file_name,label\n"
        + "".join(f"{n},{'frog' if i % 2 else 'bird'}\n" for i, n in enumerate(names))
    )
    return gt


def test_batches_and_reports_skipped_files(tmp_path) -> None:
    names = [f"{i}.wav" for i in range(5)] + ["broken.wav"]
    gt = _setup(tmp_path, names)
    batches = []

    def load(path):
        if path.endswith("broken.wav"):
            raise RuntimeError("bad header")
        return torch.zeros(1, 10), 16000

    def predict_batch(batch, model):
        batches.append(batch.shape[0])
        return ["frog"] * batch.shape[0]

    with (
        patch("bioamla.ml.ast.load_pretrained_ast_model"),
        patch("bioamla.ml.ast.get_cached_feature_extractor"),
        patch("bioamla.audio.torchaudio.load_waveform_tensor", side_effect=load),
        patch("bioamla.audio.torchaudio.resample_waveform_tensor", side_effect=lambda w, *a: w),
        patch("bioamla.ml.ast.extract_features", return_value=torch.zeros(1, 4, 2)),
        patch("bioamla.ml.ast.ast_predict_batch", side_effect=predict_batch),
    ):
        result = evaluate_directory(str(tmp_path), "m", str(gt), batch_size=2)

    assert batches == [2, 2, 1]  # --batch-size is honoured
    assert result.total_samples == 5
    assert len(result.skipped) == 1 and "broken.wav" in result.skipped[0]
    assert set(result.per_class) == {"bird", "frog"}


def test_batch_size_must_be_positive(tmp_path) -> None:
    from bioamla.exceptions import InvalidInputError

    gt = _setup(tmp_path, ["a.wav"])
    with pytest.raises(InvalidInputError):
        evaluate_directory(str(tmp_path), "m", str(gt), batch_size=0)


def test_wrong_prediction_count_is_a_model_error(tmp_path) -> None:
    from bioamla.exceptions import ModelError

    gt = _setup(tmp_path, ["a.wav", "b.wav"])
    with (
        patch("bioamla.ml.ast.load_pretrained_ast_model"),
        patch("bioamla.ml.ast.get_cached_feature_extractor"),
        patch(
            "bioamla.audio.torchaudio.load_waveform_tensor",
            return_value=(torch.zeros(1, 10), 16000),
        ),
        patch("bioamla.audio.torchaudio.resample_waveform_tensor", side_effect=lambda w, *a: w),
        patch("bioamla.ml.ast.extract_features", return_value=torch.zeros(1, 4, 2)),
        patch("bioamla.ml.ast.ast_predict_batch", return_value=[]),
    ):
        with pytest.raises(ModelError, match="returned 0 prediction"):
            evaluate_directory(str(tmp_path), "m", str(gt))


def test_mono_waveform_downmixes_channels() -> None:
    from bioamla.ml.ast import mono_waveform

    stereo = torch.stack([torch.ones(4), torch.zeros(4)])
    assert mono_waveform(stereo).tolist() == [0.5] * 4
    assert mono_waveform(torch.ones(1, 4)).shape == (4,)


def _patches(predict_batch, features=None):
    from contextlib import ExitStack

    stack = ExitStack()
    for target, kwargs in [
        ("bioamla.ml.ast.load_pretrained_ast_model", {}),
        ("bioamla.ml.ast.get_cached_feature_extractor", {}),
        (
            "bioamla.audio.torchaudio.load_waveform_tensor",
            {"return_value": (torch.zeros(1, 10), 16000)},
        ),
        ("bioamla.audio.torchaudio.resample_waveform_tensor", {"side_effect": lambda w, *a: w}),
        ("bioamla.ml.ast.extract_features", features or {"return_value": torch.zeros(1, 4, 2)}),
        ("bioamla.ml.ast.ast_predict_batch", {"side_effect": predict_batch}),
    ]:
        stack.enter_context(patch(target, **kwargs))
    return stack


def test_failed_batch_is_retried_one_file_at_a_time(tmp_path) -> None:
    from bioamla.exceptions import ModelError

    gt = _setup(tmp_path, ["a.wav", "b.wav", "c.wav"])
    calls = []

    def predict_batch(batch, model):
        calls.append(batch.shape[0])
        if batch.shape[0] > 1:
            raise ModelError("out of memory")
        if len(calls) == 3:  # the second file fails on its own too
            raise ModelError("bad input")
        return ["frog"]

    with _patches(predict_batch):
        result = evaluate_directory(str(tmp_path), "m", str(gt), batch_size=3)
    assert calls == [3, 1, 1, 1]
    assert result.total_samples == 2
    assert len(result.skipped) == 1 and "bad input" in result.skipped[0]


def test_file_giving_more_than_one_input_is_skipped(tmp_path) -> None:
    gt = _setup(tmp_path, ["a.wav", "b.wav"])
    shapes = iter([torch.zeros(2, 4, 2), torch.zeros(1, 4, 2)])
    with _patches(lambda b, m: ["frog"] * b.shape[0], {"side_effect": lambda *a: next(shapes)}):
        result = evaluate_directory(str(tmp_path), "m", str(gt))
    assert result.total_samples == 1
    assert "expected one input per file" in result.skipped[0]


def test_txt_report_has_the_confusion_matrix() -> None:
    from bioamla.cli.commands.models import _evaluation_text
    from bioamla.ml.ast_service import EvaluationResult

    m = _compute_metrics(["frog", "bird"], ["frog", "frog"])
    report = _evaluation_text(
        EvaluationResult(
            accuracy=m["accuracy"],
            precision=m["precision"],
            recall=m["recall"],
            f1_score=m["f1_score"],
            total_samples=2,
            confusion_matrix=m["confusion_matrix"],
            labels=m["labels"],
            per_class=m["per_class"],
            skipped=["x.wav: bad"],
        )
    )
    assert "confusion matrix (rows: true, columns: predicted)" in report
    assert "x.wav: bad" in report
