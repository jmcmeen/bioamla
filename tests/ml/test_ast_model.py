"""Coverage tests for bioamla.ml.ast_model (ASTModel wrapper).

Nothing hits the network: ``AutoModelForAudioClassification.from_pretrained`` and
``ASTFeatureExtractor.from_pretrained`` are patched to return lightweight fakes,
and the model's forward returns a tiny logits tensor.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bioamla.exceptions import ModelError
from bioamla.ml.ast_model import ASTModel
from bioamla.ml.base import ModelConfig


def _fake_feature_extractor():
    """A callable returning an object with ``.input_values``."""
    fe = MagicMock()

    def _call(audio, **kwargs):
        # AST features: (batch, time, mel). One frame is fine for the wrapper.
        return SimpleNamespace(input_values=torch.zeros(1, 4, 8))

    fe.side_effect = _call
    return fe


def _fake_model(n_labels=2, hidden=8):
    """A model whose __call__ returns logits / hidden_states / attentions."""
    model = MagicMock()
    model.config = SimpleNamespace(id2label={0: "a", 1: "b"}, label2id={"a": 0, "b": 1})

    def _call(input_values, output_hidden_states=False, output_attentions=False):
        logits = torch.tensor([[2.0, 1.0]][:1])  # (1, 2)
        out = SimpleNamespace(logits=logits)
        if output_hidden_states:
            out.hidden_states = [torch.ones(1, 3, hidden) for _ in range(3)]
        if output_attentions:
            out.attentions = [torch.ones(1, 2, 3, 3) for _ in range(2)]
        return out

    model.side_effect = _call
    return model


def _loaded_model(**cfg_kwargs):
    cfg = ModelConfig(device="cpu", top_k=1, **cfg_kwargs)
    m = ASTModel(config=cfg)
    fake_model = _fake_model()
    with (
        patch(
            "transformers.AutoModelForAudioClassification.from_pretrained",
            return_value=fake_model,
        ),
        patch(
            "transformers.ASTFeatureExtractor.from_pretrained",
            return_value=_fake_feature_extractor(),
        ),
    ):
        m.load("fake/model")
    return m


class TestLoad:
    def test_backend(self) -> None:
        m = ASTModel(ModelConfig(device="cpu"))
        from bioamla.ml.base import ModelBackend

        assert m.backend is ModelBackend.AST

    def test_load_sets_labels(self) -> None:
        m = _loaded_model()
        assert m.id2label == {0: "a", 1: "b"}
        assert m.label2id == {"a": 0, "b": 1}

    def test_load_failure_raises_model_error(self) -> None:
        m = ASTModel(ModelConfig(device="cpu"))
        with (
            patch(
                "transformers.AutoModelForAudioClassification.from_pretrained",
                side_effect=RuntimeError("nope"),
            ),
            patch("transformers.ASTFeatureExtractor.from_pretrained"),
        ):
            with pytest.raises(ModelError, match="Failed to load AST model"):
                m.load("fake/model")

    def test_load_feature_extractor_fallback_on_oserror(self) -> None:
        m = ASTModel(ModelConfig(device="cpu"))
        fake_fe_cls = MagicMock(return_value=_fake_feature_extractor())
        fake_fe_cls.from_pretrained.side_effect = OSError("missing")
        with (
            patch(
                "transformers.AutoModelForAudioClassification.from_pretrained",
                return_value=_fake_model(),
            ),
            patch("transformers.ASTFeatureExtractor", fake_fe_cls),
        ):
            m.load("fake/model")
        assert m.feature_extractor is not None


class TestPredict:
    def test_predict_not_loaded(self) -> None:
        m = ASTModel(ModelConfig(device="cpu"))
        with pytest.raises(ModelError, match="Model not loaded"):
            m.predict(np.zeros(16000, dtype=np.float32), sample_rate=16000)

    def test_predict_array_single_segment(self) -> None:
        m = _loaded_model(clip_duration=3.0)
        audio = np.zeros(16000, dtype=np.float32)  # 1s < clip -> single segment
        results = m.predict(audio, sample_rate=16000)
        assert len(results) == 1
        assert results[0].label == "a"  # logit 2.0 wins
        assert results[0].logits is not None  # top_k == 1

    def test_predict_min_confidence_filters(self) -> None:
        m = _loaded_model(clip_duration=3.0, min_confidence=0.99)
        results = m.predict(np.zeros(16000, dtype=np.float32), sample_rate=16000)
        # softmax([2,1]) top prob ~0.73 < 0.99 -> filtered out
        assert results == []

    def test_predict_forward_failure(self) -> None:
        m = _loaded_model(clip_duration=3.0)
        m.model.side_effect = RuntimeError("boom")
        with pytest.raises(ModelError, match="AST inference failed"):
            m.predict(np.zeros(16000, dtype=np.float32), sample_rate=16000)

    def test_predict_tensor_input_1d(self) -> None:
        m = _loaded_model(clip_duration=3.0)
        results = m.predict(torch.zeros(16000), sample_rate=16000)
        assert len(results) == 1


class TestEmbeddingsAndAttention:
    def test_extract_embeddings_default_layer(self) -> None:
        m = _loaded_model()
        emb = m.extract_embeddings(np.zeros(16000, dtype=np.float32), sample_rate=16000)
        assert emb.shape == (1, 8)

    def test_extract_embeddings_last_hidden_state(self) -> None:
        m = _loaded_model()
        emb = m.extract_embeddings(
            np.zeros(16000, dtype=np.float32), sample_rate=16000, layer="last_hidden_state"
        )
        assert emb.shape == (1, 8)

    def test_extract_embeddings_indexed_layer(self) -> None:
        m = _loaded_model()
        emb = m.extract_embeddings(
            np.zeros(16000, dtype=np.float32), sample_rate=16000, layer="layer_0"
        )
        assert emb.shape == (1, 8)

    def test_extract_embeddings_not_loaded(self) -> None:
        m = ASTModel(ModelConfig(device="cpu"))
        with pytest.raises(ModelError, match="Model not loaded"):
            m.extract_embeddings(np.zeros(16000, dtype=np.float32), sample_rate=16000)

    def test_extract_embeddings_forward_failure(self) -> None:
        m = _loaded_model()
        m.model.side_effect = RuntimeError("boom")
        with pytest.raises(ModelError, match="AST embedding extraction failed"):
            m.extract_embeddings(np.zeros(16000, dtype=np.float32), sample_rate=16000)

    def test_get_attention_weights(self) -> None:
        m = _loaded_model()
        attns = m.get_attention_weights(np.zeros(16000, dtype=np.float32), sample_rate=16000)
        assert len(attns) == 2
        assert attns[0].shape == (1, 2, 3, 3)

    def test_get_attention_not_loaded(self) -> None:
        m = ASTModel(ModelConfig(device="cpu"))
        with pytest.raises(ModelError, match="Model not loaded"):
            m.get_attention_weights(np.zeros(16000, dtype=np.float32), sample_rate=16000)

    def test_get_attention_forward_failure(self) -> None:
        m = _loaded_model()
        m.model.side_effect = RuntimeError("boom")
        with pytest.raises(ModelError, match="AST attention extraction failed"):
            m.get_attention_weights(np.zeros(16000, dtype=np.float32), sample_rate=16000)


class TestLoadWaveform:
    def test_load_waveform_str_failure(self) -> None:
        m = _loaded_model()
        with patch(
            "bioamla.audio.torchaudio.load_waveform_tensor", side_effect=RuntimeError("bad")
        ):
            with pytest.raises(ModelError, match="Failed to load audio"):
                m.predict("missing.wav")

    def test_dummy_input(self) -> None:
        m = _loaded_model()
        out = m._get_dummy_input()
        assert out.shape[0] == 1
