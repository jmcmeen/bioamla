"""bioamla.protocols: importable on the base install, and satisfied by the real implementations."""

import subprocess
import sys

import pytest

from bioamla import protocols


def test_imports_only_the_standard_library() -> None:
    probe = (
        "import sys, bioamla.protocols\n"
        "print(' '.join(m for m in ('numpy', 'torch', 'pydantic', 'bioamla.ml', 'bioamla.common')"
        " if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


def test_exports() -> None:
    assert set(protocols.__all__) == {
        "AudioModel",
        "OnBatchProgress",
        "OnDownloadProgress",
        "OnProgress",
        "OnTrainProgress",
        "ShouldStop",
    }


def test_existing_aliases_point_at_the_protocols() -> None:
    from bioamla.catalogs.inat import ProgressCallback as DownloadAlias
    from bioamla.common.progress import ProgressCallback as BatchAlias

    assert BatchAlias is protocols.OnBatchProgress
    assert DownloadAlias is protocols.OnDownloadProgress


def test_base_audio_model_subclasses_are_audio_models() -> None:
    pytest.importorskip("torch")
    from bioamla.ml.base import BaseAudioModel, ModelBackend

    class Dummy(BaseAudioModel):
        @property
        def backend(self) -> ModelBackend:
            return next(iter(ModelBackend))

        def load(self, model_path, **kwargs):
            return self

        def predict(self, audio, sample_rate=None):
            return []

        def extract_embeddings(self, audio, sample_rate=None, layer=None):
            return None

    assert isinstance(Dummy(), protocols.AudioModel)


def test_object_missing_members_is_not_an_audio_model() -> None:
    class NotAModel:
        def predict(self, audio, sample_rate=None):
            return []

    assert not isinstance(NotAModel(), protocols.AudioModel)
