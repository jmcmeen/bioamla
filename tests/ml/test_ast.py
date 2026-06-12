"""Coverage tests for bioamla.ml.ast (low-level AST helpers).

transformers/torch model construction and forward are patched so nothing is
downloaded. Covers InferenceConfig, cached feature extractor, model load
(local/hub/fp16/compile/failure), feature extraction, predict / predict_batch,
and wav_ast_inference.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

from bioamla.exceptions import ModelError
from bioamla.ml import ast as ast_mod
from bioamla.ml.ast import (
    InferenceConfig,
    ast_predict,
    ast_predict_batch,
    extract_features,
    get_cached_feature_extractor,
    load_pretrained_ast_model,
    wav_ast_inference,
)


def _fake_model():
    model = MagicMock()
    model.config = SimpleNamespace(id2label={0: "a", 1: "b"})
    model.side_effect = lambda iv, **kw: SimpleNamespace(logits=torch.tensor([[1.0, 5.0]]))
    return model


class TestInferenceConfig:
    def test_defaults(self) -> None:
        c = InferenceConfig()
        assert c.batch_size == 8
        assert c.use_fp16 is False
        assert c.use_compile is False
        assert c.num_workers == 1


class TestCachedFeatureExtractor:
    def test_no_path_returns_default(self) -> None:
        get_cached_feature_extractor.cache_clear()
        fake = MagicMock()
        with patch("transformers.ASTFeatureExtractor", return_value=fake):
            fe = get_cached_feature_extractor(None)
        assert fe is fake

    def test_path_oserror_falls_back(self) -> None:
        get_cached_feature_extractor.cache_clear()
        default = MagicMock()
        fake_cls = MagicMock(return_value=default)
        fake_cls.from_pretrained.side_effect = OSError("missing")
        with patch("transformers.ASTFeatureExtractor", fake_cls):
            fe = get_cached_feature_extractor("some/path")
        assert fe is default

    def test_path_loads(self) -> None:
        get_cached_feature_extractor.cache_clear()
        loaded = MagicMock()
        with patch("transformers.ASTFeatureExtractor.from_pretrained", return_value=loaded):
            fe = get_cached_feature_extractor("good/path2")
        assert fe is loaded


class TestLoadPretrained:
    def test_hub_load(self) -> None:
        fake = _fake_model()
        with patch(
            "transformers.AutoModelForAudioClassification.from_pretrained", return_value=fake
        ) as fp:
            out = load_pretrained_ast_model("MIT/some-model")
        assert out is fake
        # hub path: no local_files_only kwarg
        assert "local_files_only" not in fp.call_args.kwargs

    def test_local_load(self) -> None:
        fake = _fake_model()
        with patch(
            "transformers.AutoModelForAudioClassification.from_pretrained", return_value=fake
        ) as fp:
            load_pretrained_ast_model("./local-model")
        assert fp.call_args.kwargs.get("local_files_only") is True

    def test_fp16(self) -> None:
        fake = _fake_model()
        with patch(
            "transformers.AutoModelForAudioClassification.from_pretrained", return_value=fake
        ) as fp:
            load_pretrained_ast_model("hub/model", use_fp16=True)
        assert fp.call_args.kwargs.get("torch_dtype") == torch.float16

    def test_compile(self) -> None:
        fake = _fake_model()
        compiled = MagicMock()
        with (
            patch(
                "transformers.AutoModelForAudioClassification.from_pretrained", return_value=fake
            ),
            patch.object(torch, "compile", return_value=compiled) as comp,
        ):
            out = load_pretrained_ast_model("hub/model", use_compile=True)
        comp.assert_called_once()
        assert out is compiled

    def test_failure_raises(self) -> None:
        with patch(
            "transformers.AutoModelForAudioClassification.from_pretrained",
            side_effect=RuntimeError("nope"),
        ):
            with pytest.raises(ModelError, match="Failed to load AST model"):
                load_pretrained_ast_model("hub/model")


class TestExtractFeatures:
    def test_with_provided_extractor(self) -> None:
        fe = MagicMock()
        fe.return_value = SimpleNamespace(input_values=torch.zeros(1, 4, 8))
        fe.return_value.to = lambda d: fe.return_value
        # feature_extractor(...) returns obj; code calls .to(device) on it.
        fe.side_effect = lambda *a, **k: SimpleNamespace(
            input_values=torch.zeros(1, 4, 8),
            to=lambda d: SimpleNamespace(input_values=torch.zeros(1, 4, 8)),
        )
        out = extract_features(
            torch.zeros(1, 16000), 16000, feature_extractor=fe, device=torch.device("cpu")
        )
        assert out.shape == (1, 4, 8)

    def test_default_extractor_and_device(self) -> None:
        fe_obj = SimpleNamespace(
            input_values=torch.zeros(1, 4, 8),
        )
        fe = MagicMock()
        fe.side_effect = lambda *a, **k: SimpleNamespace(
            input_values=torch.zeros(1, 4, 8), to=lambda d: fe_obj
        )
        with (
            patch.object(ast_mod, "get_cached_feature_extractor", return_value=fe),
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            out = extract_features(torch.zeros(1, 16000), 16000)
        assert out.shape == (1, 4, 8)


class TestPredictFns:
    def test_ast_predict(self) -> None:
        model = _fake_model()
        label = ast_predict(torch.zeros(1, 4, 8), model)
        assert label == "b"  # logit 5.0 wins

    def test_ast_predict_failure(self) -> None:
        model = _fake_model()
        model.side_effect = RuntimeError("boom")
        with pytest.raises(ModelError, match="AST prediction failed"):
            ast_predict(torch.zeros(1, 4, 8), model)

    def test_ast_predict_batch(self) -> None:
        model = MagicMock()
        model.config = SimpleNamespace(id2label={0: "a", 1: "b"})
        model.side_effect = lambda iv, **kw: SimpleNamespace(
            logits=torch.tensor([[1.0, 5.0], [9.0, 0.0]])
        )
        labels = ast_predict_batch(torch.zeros(2, 4, 8), model)
        assert labels == ["b", "a"]

    def test_ast_predict_batch_failure(self) -> None:
        model = _fake_model()
        model.side_effect = RuntimeError("boom")
        with pytest.raises(ModelError, match="AST batch prediction failed"):
            ast_predict_batch(torch.zeros(2, 4, 8), model)


class TestWavAstInference:
    def test_happy_path(self) -> None:
        model = _fake_model()
        fe = MagicMock()
        with (
            patch(
                "bioamla.audio.torchaudio.load_waveform_tensor",
                return_value=(torch.zeros(1, 16000), 16000),
            ),
            patch(
                "bioamla.audio.torchaudio.resample_waveform_tensor",
                return_value=torch.zeros(1, 16000),
            ),
            patch.object(ast_mod, "get_cached_feature_extractor", return_value=fe),
            patch.object(ast_mod, "extract_features", return_value=torch.zeros(1, 4, 8)),
            patch.object(ast_mod, "load_pretrained_ast_model", return_value=model),
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            label = wav_ast_inference("a.wav", "hub/model", 16000)
        assert label == "b"

    def test_load_failure(self) -> None:
        with patch(
            "bioamla.audio.torchaudio.load_waveform_tensor", side_effect=RuntimeError("bad")
        ):
            with pytest.raises(ModelError, match="Failed to load audio"):
                wav_ast_inference("a.wav", "hub/model", 16000)
