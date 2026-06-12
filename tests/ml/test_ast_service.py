"""Coverage tests for bioamla.ml.ast_service.

Single-file predict, embedding extraction (AutoModel CLS token), directory
evaluation, and the private metrics/ground-truth helpers. All transformers/torch
model construction is patched so nothing downloads.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from bioamla.exceptions import InvalidInputError, ModelError, NotFoundError
from bioamla.ml.ast_service import (
    EvaluationResult,
    TrainResult,
    _compute_metrics,
    _load_ground_truth,
    evaluate_directory,
    extract_embeddings_file,
    get_model_info,
    predict_file,
)


class TestDataclasses:
    def test_evaluation_to_dict_with_cm(self) -> None:
        r = EvaluationResult(
            accuracy=1.0,
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            total_samples=2,
            confusion_matrix=[[1, 0], [0, 1]],
        )
        d = r.to_dict()
        assert d["confusion_matrix"] == [[1, 0], [0, 1]]

    def test_train_to_dict(self) -> None:
        d = TrainResult(model_path="/m", epochs=2, final_accuracy=0.5, final_loss=0.1).to_dict()
        assert d["final_accuracy"] == 0.5
        assert d["final_loss"] == 0.1


class TestPredictFile:
    def test_missing_file(self) -> None:
        with pytest.raises(NotFoundError, match="not found"):
            predict_file("/no/such.wav")

    def test_delegates_to_inference(self, test_audio_path) -> None:
        fake_inf = MagicMock()
        fake_inf.predict.return_value = "RESULT"
        with patch("bioamla.ml.inference.ASTInference", return_value=fake_inf) as AI:
            out = predict_file(test_audio_path, model_path="hub/model")
        assert out == "RESULT"
        AI.assert_called_once()
        fake_inf.predict.assert_called_once_with(test_audio_path)


class TestExtractEmbeddingsFile:
    def test_missing_file(self) -> None:
        with pytest.raises(NotFoundError, match="not found"):
            extract_embeddings_file("/no/such.wav", model_path="hub/model")

    @pytest.mark.usefixtures("requires_torchcodec")
    def test_happy_path(self, test_audio_path) -> None:
        torch = pytest.importorskip("torch")
        fake_model = MagicMock()
        fake_model.to.return_value = fake_model
        hidden = torch.ones(1, 5, 8)
        fake_model.side_effect = lambda **kw: SimpleNamespace(last_hidden_state=hidden)
        fe = MagicMock()
        fe.return_value = SimpleNamespace()
        # feature_extractor(...).to(device) -> dict-like consumed via model(**inputs)
        fe.side_effect = lambda *a, **k: SimpleNamespace(
            to=lambda d: {"input_values": torch.zeros(1, 4, 8)}
        )

        with (
            patch("transformers.ASTFeatureExtractor.from_pretrained", return_value=fe),
            patch("transformers.AutoModel.from_pretrained", return_value=fake_model),
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            out = extract_embeddings_file(test_audio_path, model_path="hub/model")
        assert out["filepath"] == test_audio_path
        assert out["shape"] == (1, 8)
        assert "float" in out["dtype"]

    @pytest.mark.usefixtures("requires_torchcodec")
    def test_model_load_failure(self, test_audio_path) -> None:
        torch = pytest.importorskip("torch")
        fe = MagicMock()
        with (
            patch("transformers.ASTFeatureExtractor.from_pretrained", return_value=fe),
            patch("transformers.AutoModel.from_pretrained", side_effect=RuntimeError("nope")),
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            with pytest.raises(ModelError, match="Failed to load AST model"):
                extract_embeddings_file(test_audio_path, model_path="hub/model")


class TestGetModelInfo:
    def test_delegates(self) -> None:
        with patch("bioamla.ml.embedding.get_ast_model_info", return_value={"num_classes": 3}) as g:
            out = get_model_info("hub/model")
        assert out == {"num_classes": 3}
        g.assert_called_once_with("hub/model")


class TestLoadGroundTruth:
    def test_missing_file_column(self, tmp_path) -> None:
        csv = tmp_path / "gt.csv"
        csv.write_text("label\nfrog\n")
        with pytest.raises(InvalidInputError, match="file_name"):
            _load_ground_truth(str(csv), "file_name", "label")

    def test_missing_label_column(self, tmp_path) -> None:
        csv = tmp_path / "gt.csv"
        csv.write_text("file_name\na.wav\n")
        with pytest.raises(InvalidInputError, match="label"):
            _load_ground_truth(str(csv), "file_name", "label")

    def test_keyed_by_basename(self, tmp_path) -> None:
        csv = tmp_path / "gt.csv"
        csv.write_text("file_name,label\nsub/a.wav,frog\nb.wav,bird\n")
        gt = _load_ground_truth(str(csv), "file_name", "label")
        assert gt == {"a.wav": "frog", "b.wav": "bird"}


class TestComputeMetrics:
    def test_perfect(self) -> None:
        m = _compute_metrics(["a", "b", "a"], ["a", "b", "a"])
        assert m["accuracy"] == 1.0
        assert m["correct_predictions"] == 3
        assert m["total_samples"] == 3

    def test_mixed(self) -> None:
        m = _compute_metrics(["a", "b"], ["a", "a"])
        assert m["accuracy"] == 0.5

    def test_length_mismatch(self) -> None:
        with pytest.raises(InvalidInputError, match="Length mismatch"):
            _compute_metrics(["a"], ["a", "b"])


class TestEvaluateDirectory:
    def test_missing_audio_dir(self, tmp_path) -> None:
        gt = tmp_path / "gt.csv"
        gt.write_text("file_name,label\na.wav,frog\n")
        with pytest.raises(NotFoundError, match="Audio directory not found"):
            evaluate_directory("/no/dir", "hub/model", str(gt))

    def test_missing_csv(self, test_audio_dir) -> None:
        with pytest.raises(NotFoundError, match="Ground truth file not found"):
            evaluate_directory(test_audio_dir, "hub/model", "/no/gt.csv")

    def test_no_matching_files(self, test_audio_dir, tmp_path) -> None:
        torch = pytest.importorskip("torch")
        gt = tmp_path / "gt.csv"
        gt.write_text("file_name,label\nnope.wav,frog\n")
        with patch.object(torch.cuda, "is_available", return_value=False):
            with pytest.raises(NotFoundError, match="match ground truth"):
                evaluate_directory(test_audio_dir, "hub/model", str(gt))

    def test_full_evaluation(self, test_audio_dir, tmp_path) -> None:
        torch = pytest.importorskip("torch")
        # ground truth covering the three fixture files
        gt = tmp_path / "gt.csv"
        gt.write_text("file_name,label\naudio_0.wav,a\naudio_1.wav,b\naudio_2.wav,a\n")
        fake_model = MagicMock()
        with (
            patch("bioamla.ml.ast.load_pretrained_ast_model", return_value=fake_model),
            patch("bioamla.ml.ast.get_cached_feature_extractor", return_value=MagicMock()),
            patch(
                "bioamla.audio.torchaudio.load_waveform_tensor",
                return_value=(torch.zeros(1, 16000), 16000),
            ),
            patch(
                "bioamla.audio.torchaudio.resample_waveform_tensor",
                return_value=torch.zeros(1, 16000),
            ),
            patch("bioamla.ml.ast.extract_features", return_value=torch.zeros(1, 4, 8)),
            patch("bioamla.ml.ast.ast_predict", return_value="a"),
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            result = evaluate_directory(test_audio_dir, "hub/model", str(gt))
        assert result.total_samples == 3
        assert isinstance(result, EvaluationResult)

    def test_no_predictions_raises(self, test_audio_dir, tmp_path) -> None:
        torch = pytest.importorskip("torch")
        gt = tmp_path / "gt.csv"
        gt.write_text("file_name,label\naudio_0.wav,a\n")
        with (
            patch("bioamla.ml.ast.load_pretrained_ast_model", return_value=MagicMock()),
            patch("bioamla.ml.ast.get_cached_feature_extractor", return_value=MagicMock()),
            patch("bioamla.audio.torchaudio.load_waveform_tensor", side_effect=RuntimeError("bad")),
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            with pytest.raises(ModelError, match="No predictions were generated"):
                evaluate_directory(test_audio_dir, "hub/model", str(gt))
