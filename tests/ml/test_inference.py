"""Coverage tests for bioamla.ml.inference (ASTInference).

The model and feature extractor are patched at their import site
(``transformers.*``) so nothing is downloaded; the model forward returns a tiny
logits tensor and the torchaudio loaders return synthetic waveforms.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

from bioamla.exceptions import ModelError
from bioamla.ml.inference import ASTInference, ASTPredictionResult


def _fake_feature_extractor():
    fe = MagicMock()
    fe.side_effect = lambda audio, **kw: SimpleNamespace(input_values=torch.zeros(1, 4, 8))
    return fe


def _fake_model(n_labels=3):
    model = MagicMock()
    model.config = SimpleNamespace(
        id2label={0: "a", 1: "b", 2: "c"}, label2id={"a": 0, "b": 1, "c": 2}
    )
    model.side_effect = lambda iv, **kw: SimpleNamespace(logits=torch.tensor([[3.0, 1.0, 0.5]]))
    return model


def _make_inference():
    with (
        patch(
            "transformers.AutoModelForAudioClassification.from_pretrained",
            return_value=_fake_model(),
        ),
        patch(
            "transformers.ASTFeatureExtractor.from_pretrained",
            return_value=_fake_feature_extractor(),
        ),
        patch("bioamla.ml.device.get_device", return_value=torch.device("cpu")),
    ):
        return ASTInference(model_path="fake/model", sample_rate=16000)


class TestInit:
    def test_init_loads_labels(self) -> None:
        inf = _make_inference()
        assert len(inf.id2label) == 3
        assert inf.device.type == "cpu"

    def test_init_failure_raises(self) -> None:
        with (
            patch(
                "transformers.AutoModelForAudioClassification.from_pretrained",
                side_effect=RuntimeError("nope"),
            ),
            patch("transformers.ASTFeatureExtractor.from_pretrained"),
            patch("bioamla.ml.device.get_device", return_value=torch.device("cpu")),
        ):
            with pytest.raises(ModelError, match="Failed to load AST model"):
                ASTInference(model_path="fake/model")

    def test_init_string_device(self) -> None:
        with (
            patch(
                "transformers.AutoModelForAudioClassification.from_pretrained",
                return_value=_fake_model(),
            ),
            patch(
                "transformers.ASTFeatureExtractor.from_pretrained",
                return_value=_fake_feature_extractor(),
            ),
        ):
            inf = ASTInference(model_path="fake/model", device="cpu")
        assert inf.device.type == "cpu"


def _patch_audio(waveform=None, orig_sr=16000):
    if waveform is None:
        waveform = torch.zeros(1, 16000)
    return patch("bioamla.audio.torchaudio.load_waveform_tensor", return_value=(waveform, orig_sr))


class TestPredict:
    def test_predict_basic(self) -> None:
        inf = _make_inference()
        with _patch_audio():
            res = inf.predict("a.wav")
        assert isinstance(res, ASTPredictionResult)
        assert res.predicted_label == "a"
        assert res.logits is None

    def test_predict_return_logits(self) -> None:
        inf = _make_inference()
        with _patch_audio():
            res = inf.predict("a.wav", return_logits=True)
        assert res.logits is not None and len(res.logits) == 3

    def test_predict_resamples(self) -> None:
        inf = _make_inference()
        with (
            _patch_audio(orig_sr=44100),
            patch(
                "bioamla.audio.torchaudio.resample_waveform_tensor",
                return_value=torch.zeros(1, 16000),
            ) as rs,
        ):
            inf.predict("a.wav")
        rs.assert_called_once()

    def test_predict_load_failure(self) -> None:
        inf = _make_inference()
        with patch(
            "bioamla.audio.torchaudio.load_waveform_tensor", side_effect=RuntimeError("bad")
        ):
            with pytest.raises(ModelError, match="Failed to load audio"):
                inf.predict("a.wav")

    def test_predict_inference_failure(self) -> None:
        inf = _make_inference()
        inf.model.side_effect = RuntimeError("boom")
        with _patch_audio():
            with pytest.raises(ModelError, match="AST inference failed"):
                inf.predict("a.wav")


class TestPredictTopk:
    def test_topk_basic(self) -> None:
        inf = _make_inference()
        with _patch_audio():
            res = inf.predict_topk("a.wav", top_k=2)
        assert res.predicted_label == "a"
        assert res.top_k_labels == ["a", "b"]
        assert len(res.top_k_scores) == 2

    def test_topk_min_confidence_drops(self) -> None:
        inf = _make_inference()
        with _patch_audio():
            res = inf.predict_topk("a.wav", top_k=3, min_confidence=0.5)
        # softmax([3,1,0.5]) -> top ~0.84, rest small -> only first kept
        assert res.top_k_labels == ["a"]

    def test_topk_clamped(self) -> None:
        inf = _make_inference()
        with _patch_audio():
            res = inf.predict_topk("a.wav", top_k=99)
        assert len(res.top_k_labels) == 3

    def test_topk_load_failure(self) -> None:
        inf = _make_inference()
        with patch(
            "bioamla.audio.torchaudio.load_waveform_tensor", side_effect=RuntimeError("bad")
        ):
            with pytest.raises(ModelError, match="Failed to load audio"):
                inf.predict_topk("a.wav")

    def test_topk_inference_failure(self) -> None:
        inf = _make_inference()
        inf.model.side_effect = RuntimeError("boom")
        with _patch_audio():
            with pytest.raises(ModelError, match="AST inference failed"):
                inf.predict_topk("a.wav")


class TestPredictSegments:
    def test_segments_basic(self) -> None:
        inf = _make_inference()
        # 6s waveform, 3s segments -> 2 segments
        wav = torch.zeros(1, 16000 * 6)
        with _patch_audio(waveform=wav):
            results = inf.predict_segments("a.wav", clip_length=3, overlap=0)
        assert len(results) == 2
        assert all(r.predicted_label == "a" for r in results)

    def test_segments_return_logits(self) -> None:
        inf = _make_inference()
        wav = torch.zeros(1, 16000 * 3)
        with _patch_audio(waveform=wav):
            results = inf.predict_segments("a.wav", clip_length=3, return_logits=True)
        assert results[0].logits is not None

    def test_segments_load_failure(self) -> None:
        inf = _make_inference()
        with patch(
            "bioamla.audio.torchaudio.load_waveform_tensor", side_effect=RuntimeError("bad")
        ):
            with pytest.raises(ModelError, match="Failed to load audio"):
                inf.predict_segments("a.wav")

    def test_segments_inference_failure(self) -> None:
        inf = _make_inference()
        inf.model.side_effect = RuntimeError("boom")
        wav = torch.zeros(1, 16000 * 3)
        with _patch_audio(waveform=wav):
            with pytest.raises(ModelError, match="AST inference failed"):
                inf.predict_segments("a.wav", clip_length=3)
