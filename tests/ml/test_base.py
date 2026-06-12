"""Coverage tests for bioamla.ml.base.

Exercises the concrete dataclasses, ModelConfig device selection, the
BaseAudioModel non-abstract helpers (batch/filter/save/to/eval/repr) via a tiny
concrete subclass, and the model registry. torch is mocked where heavy.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bioamla.exceptions import InvalidInputError, ModelError
from bioamla.ml import base
from bioamla.ml.base import (
    BaseAudioModel,
    BatchPredictionResult,
    ModelBackend,
    ModelConfig,
    PredictionResult,
    get_model_class,
    list_models,
    register_model,
)


class _ToyModel(BaseAudioModel):
    """Minimal concrete model that returns canned predictions."""

    @property
    def backend(self) -> ModelBackend:
        return ModelBackend.AST

    def load(self, model_path: str, **kwargs):
        self.id2label = {0: "a", 1: "b"}
        self.label2id = {"a": 0, "b": 1}
        return self

    def predict(self, audio, sample_rate=None):
        if audio == "BOOM":
            raise ModelError("boom")
        return [PredictionResult(label="a", confidence=0.9, filepath=str(audio))]

    def extract_embeddings(self, audio, sample_rate=None, layer=None):
        return np.ones(4, dtype=np.float32)


class TestDataclasses:
    def test_prediction_result_to_dict_minimal(self) -> None:
        p = PredictionResult(label="frog", confidence=0.5)
        d = p.to_dict()
        assert d["label"] == "frog"
        assert "filepath" not in d
        assert "metadata" not in d

    def test_prediction_result_to_dict_full(self) -> None:
        p = PredictionResult(label="frog", confidence=0.5, filepath="a.wav", metadata={"k": 1})
        d = p.to_dict()
        assert d["filepath"] == "a.wav"
        assert d["metadata"] == {"k": 1}

    def test_batch_prediction_result_to_dict(self) -> None:
        b = BatchPredictionResult(
            predictions=[PredictionResult(label="x", confidence=1.0)],
            total_files=1,
            files_processed=1,
            files_failed=0,
            processing_time=0.1,
        )
        d = b.to_dict()
        assert d["total_files"] == 1
        assert len(d["predictions"]) == 1


class TestModelConfig:
    def test_get_device_explicit(self) -> None:
        cfg = ModelConfig(device="cpu")
        assert cfg.get_device().type == "cpu"

    def test_get_device_auto_cpu(self) -> None:
        cfg = ModelConfig()
        with patch.object(torch.cuda, "is_available", return_value=False):
            assert cfg.get_device().type == "cpu"


class TestBaseAudioModelHelpers:
    def _model(self) -> _ToyModel:
        m = _ToyModel(ModelConfig(device="cpu"))
        m.load("x")
        return m

    def test_num_classes_and_classes(self) -> None:
        m = self._model()
        assert m.num_classes == 2
        assert set(m.classes) == {"a", "b"}

    def test_predict_file_delegates(self) -> None:
        m = self._model()
        res = m.predict_file("a.wav")
        assert res[0].label == "a"

    def test_predict_batch_counts(self) -> None:
        m = self._model()
        seen = []
        result = m.predict_batch(
            ["a.wav", "BOOM", "b.wav"], progress_callback=lambda c, t: seen.append((c, t))
        )
        assert result.total_files == 3
        assert result.files_processed == 2
        assert result.files_failed == 1
        assert seen[-1] == (3, 3)

    def test_filter_predictions(self) -> None:
        m = self._model()
        preds = [
            PredictionResult(label="a", confidence=0.9),
            PredictionResult(label="b", confidence=0.1),
            PredictionResult(label="c", confidence=0.95),
        ]
        out = m.filter_predictions(preds, min_confidence=0.5)
        assert {p.label for p in out} == {"a", "c"}

        out2 = m.filter_predictions(preds, labels=["a"])
        assert [p.label for p in out2] == ["a"]

        out3 = m.filter_predictions(preds, exclude_labels=["a"])
        assert "a" not in {p.label for p in out3}

    def test_repr(self) -> None:
        m = self._model()
        r = repr(m)
        assert "ast" in r and "classes=2" in r

    def test_to_and_eval(self) -> None:
        m = self._model()
        m.model = MagicMock()
        assert m.to("cpu") is m
        m.model.to.assert_called()
        assert m.eval() is m
        m.model.eval.assert_called()

    def test_eval_no_model(self) -> None:
        m = self._model()
        m.model = None
        assert m.eval() is m  # no-op, no error

    def test_save_unsupported_format(self, tmp_path) -> None:
        m = self._model()
        with pytest.raises(InvalidInputError, match="Unsupported format"):
            m.save(str(tmp_path / "x.bin"), format="bogus")

    def test_save_pytorch_no_model_raises(self, tmp_path) -> None:
        m = self._model()
        m.model = None
        with pytest.raises(ModelError, match="No model loaded"):
            m.save(str(tmp_path / "m.pt"), format="pt")

    def test_save_pytorch_roundtrip(self, tmp_path) -> None:
        m = self._model()
        fake = MagicMock()
        fake.state_dict.return_value = {"w": 1}
        m.model = fake
        out = tmp_path / "sub" / "m.pt"
        with patch.object(torch, "save") as tsave:
            path = m.save(str(out), format="pt")
        assert path == str(out)
        assert out.parent.exists()
        tsave.assert_called_once()
        state = tsave.call_args[0][0]
        assert state["id2label"] == {0: "a", 1: "b"}
        assert state["config"]["backend"] == "ast"


class TestModelRegistry:
    def test_register_and_get(self) -> None:
        @register_model("toy_reg_test")
        class _X:
            pass

        assert get_model_class("toy_reg_test") is _X
        assert "toy_reg_test" in list_models()

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(InvalidInputError, match="Unknown model"):
            get_model_class("definitely-not-registered")


class TestAudioDatasetFactory:
    def test_audio_dataset_is_torch_dataset(self) -> None:
        from torch.utils.data import Dataset

        ds = base.AudioDataset(["a.wav", "b.wav"], sample_rate=16000, clip_duration=3.0)
        assert isinstance(ds, Dataset)
        assert len(ds) == 2

    def test_audio_dataset_getitem_pads_and_truncates(self) -> None:
        ds = base.AudioDataset(["a.wav"], sample_rate=100, clip_duration=1.0)
        # Patch the torchaudio helpers used inside __getitem__.
        short = torch.zeros(1, 50)  # shorter than target (100) -> padded
        with (
            patch("bioamla.audio.torchaudio.load_waveform_tensor", return_value=(short, 100)),
            patch(
                "bioamla.audio.torchaudio.resample_waveform_tensor", side_effect=lambda w, a, b: w
            ),
        ):
            wav, fp = ds[0]
        assert wav.shape[0] == 100
        assert fp == "a.wav"

    def test_create_dataloader(self) -> None:
        from torch.utils.data import DataLoader

        cfg = ModelConfig(device="cpu", batch_size=2, num_workers=0)
        with patch.object(torch.cuda, "is_available", return_value=False):
            dl = base.create_dataloader(["a.wav", "b.wav"], cfg)
        assert isinstance(dl, DataLoader)
