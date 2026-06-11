"""Coverage tests for bioamla.audio.data (AudioData DTO + ToDictMixin)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from bioamla.audio import AudioData
from bioamla.audio.data import AudioMetadata, ProcessedAudio, ToDictMixin


class TestAudioDataProperties:
    def test_duration(self, sample_audio_data: AudioData) -> None:
        assert sample_audio_data.duration == pytest.approx(1.0, rel=1e-3)

    def test_num_samples(self, sample_audio_data: AudioData) -> None:
        assert sample_audio_data.num_samples == len(sample_audio_data.samples)

    def test_copy_is_deep(self, sample_audio_data: AudioData) -> None:
        clone = sample_audio_data.copy()
        clone.samples[0] = 999.0
        clone.metadata["new"] = True
        assert sample_audio_data.samples[0] != 999.0
        assert "new" not in sample_audio_data.metadata

    def test_mark_modified(self, sample_audio_data: AudioData) -> None:
        modified = sample_audio_data.mark_modified()
        assert modified.is_modified is True
        assert sample_audio_data.is_modified is False


class TestAudioDataToDict:
    def test_to_dict_serializes(self, sample_audio_data: AudioData) -> None:
        d = sample_audio_data.to_dict()
        assert d["sample_rate"] == 16000
        assert d["channels"] == 1
        assert d["source_path"] == "/test/audio.wav"
        # samples is a numpy array -> falls back to str()
        assert isinstance(d["samples"], str)
        assert d["metadata"] == {"test": True}


class TestToDictMixinValues:
    def test_path_serialized(self) -> None:
        meta = AudioMetadata(filepath="/x.wav", duration_seconds=1.0, sample_rate=16000, channels=1)
        d = meta.to_dict()
        assert d["filepath"] == "/x.wav"
        assert d["bit_depth"] is None

    def test_nested_dataclass_and_list(self) -> None:
        @dataclass
        class Inner(ToDictMixin):
            value: int

        @dataclass
        class Outer(ToDictMixin):
            inner: Inner
            items: list
            path: Path
            mapping: dict

        out = Outer(
            inner=Inner(value=5),
            items=[1, "a", (2, 3)],
            path=Path("/tmp/x"),
            mapping={"k": Inner(value=9)},
        )
        d = out.to_dict()
        assert d["inner"] == {"value": 5}
        assert d["items"] == [1, "a", [2, 3]]
        assert d["path"] == "/tmp/x"
        assert d["mapping"]["k"] == {"value": 9}

    def test_to_dict_extra_override(self) -> None:
        @dataclass
        class WithExtra(ToDictMixin):
            value: int

            def _to_dict_extra(self):
                return {"derived": self.value * 2}

        d = WithExtra(value=3).to_dict()
        assert d["derived"] == 6
        assert d["value"] == 3

    def test_non_dataclass_raises(self) -> None:
        class NotADataclass(ToDictMixin):
            pass

        with pytest.raises(TypeError):
            NotADataclass().to_dict()

    def test_none_value(self) -> None:
        @dataclass
        class Maybe(ToDictMixin):
            value: object

        assert Maybe(value=None).to_dict()["value"] is None

    def test_plain_asdict_dataclass(self) -> None:
        # A nested plain dataclass (no to_dict) hits the is_dataclass(value) branch
        @dataclass
        class Plain:
            x: int

        @dataclass
        class Holder(ToDictMixin):
            plain: object

        d = Holder(plain=Plain(x=7)).to_dict()
        assert d["plain"] == {"x": 7}

    def test_fallback_to_str(self) -> None:
        @dataclass
        class Holder(ToDictMixin):
            value: object

        arr = np.array([1, 2, 3])
        d = Holder(value=arr).to_dict()
        assert isinstance(d["value"], str)


class TestProcessedAudio:
    def test_to_dict(self) -> None:
        pa = ProcessedAudio(
            input_path="/in.wav",
            output_path="/out.wav",
            operation="resample",
            sample_rate=16000,
            duration_seconds=2.0,
        )
        d = pa.to_dict()
        assert d["operation"] == "resample"
        assert d["duration_seconds"] == 2.0
