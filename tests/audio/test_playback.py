"""Coverage tests for bioamla.audio.playback (sounddevice mocked)."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import bioamla.audio.playback as playback_mod
from bioamla.audio.playback import (
    AudioPlayer,
    PlaybackPosition,
    PlaybackState,
    play_audio,
    stop_audio,
)
from bioamla.exceptions import InvalidInputError


@pytest.fixture
def mock_sd():
    """Install a fake sounddevice module so nothing actually plays."""
    fake = MagicMock(name="sounddevice")
    fake.CallbackStop = type("CallbackStop", (Exception,), {})
    stream = MagicMock(name="OutputStream")
    fake.OutputStream.return_value = stream
    with patch.dict(sys.modules, {"sounddevice": fake}):
        yield fake, stream


@pytest.fixture
def mono_audio() -> np.ndarray:
    return np.linspace(-0.5, 0.5, 16000, dtype=np.float32)


class TestLoadAndState:
    def test_initial_state(self) -> None:
        player = AudioPlayer()
        assert player.is_stopped
        assert player.state == PlaybackState.STOPPED
        assert player.duration == 0.0
        pos = player.position
        assert isinstance(pos, PlaybackPosition)
        assert pos.total_samples == 0

    def test_load_mono(self, mono_audio) -> None:
        player = AudioPlayer()
        player.load(mono_audio, 16000)
        assert player.duration == pytest.approx(1.0, rel=1e-3)
        assert player.position.total_samples == 16000

    def test_load_stereo_channels_last(self) -> None:
        player = AudioPlayer()
        stereo = np.zeros((16000, 2), dtype=np.float32)
        player.load(stereo, 16000)
        assert player._audio.shape == (16000, 2)

    def test_load_stereo_transposed(self) -> None:
        player = AudioPlayer()
        stereo = np.zeros((2, 16000), dtype=np.float32)  # channels, samples
        player.load(stereo, 16000)
        assert player._audio.shape == (16000, 2)

    def test_load_bad_shape_raises(self) -> None:
        player = AudioPlayer()
        with pytest.raises(InvalidInputError):
            player.load(np.zeros((2, 2, 2), dtype=np.float32), 16000)


class TestPlayPauseStop:
    def test_play_without_load_raises(self, mock_sd) -> None:
        player = AudioPlayer()
        with pytest.raises(InvalidInputError):
            player.play()

    def test_play_starts_stream(self, mock_sd, mono_audio) -> None:
        fake, stream = mock_sd
        player = AudioPlayer()
        player.load(mono_audio, 16000)
        player.play()
        assert player.is_playing
        fake.OutputStream.assert_called_once()
        stream.start.assert_called_once()

    def test_play_when_already_playing_noop(self, mock_sd, mono_audio) -> None:
        fake, _ = mock_sd
        player = AudioPlayer()
        player.load(mono_audio, 16000)
        player.play()
        player.play()
        assert fake.OutputStream.call_count == 1

    def test_pause_stops_stream(self, mock_sd, mono_audio) -> None:
        _, stream = mock_sd
        player = AudioPlayer()
        player.load(mono_audio, 16000)
        player.play()
        player.pause()
        assert player.is_paused
        stream.stop.assert_called_once()
        stream.close.assert_called_once()

    def test_pause_when_not_playing_noop(self, mono_audio) -> None:
        player = AudioPlayer()
        player.load(mono_audio, 16000)
        player.pause()
        assert player.is_stopped

    def test_stop_resets(self, mock_sd, mono_audio) -> None:
        player = AudioPlayer()
        player.load(mono_audio, 16000)
        player.play()
        player.seek(0.5)
        player.stop()
        assert player.is_stopped
        assert player.position.current_sample == 0

    def test_play_stereo_channels(self, mock_sd) -> None:
        fake, _ = mock_sd
        player = AudioPlayer()
        player.load(np.zeros((16000, 2), dtype=np.float32), 16000)
        player.play()
        _, kwargs = fake.OutputStream.call_args
        assert kwargs["channels"] == 2


class TestSeek:
    def test_seek_by_time(self, mono_audio) -> None:
        player = AudioPlayer()
        player.load(mono_audio, 16000)
        player.seek(0.5)
        assert player.position.current_sample == 8000

    def test_seek_by_sample(self, mono_audio) -> None:
        player = AudioPlayer()
        player.load(mono_audio, 16000)
        player.seek(1000, by_sample=True)
        assert player.position.current_sample == 1000

    def test_seek_clamps(self, mono_audio) -> None:
        player = AudioPlayer()
        player.load(mono_audio, 16000)
        player.seek(100.0)  # past end
        assert player.position.current_sample == len(mono_audio)
        player.seek(-5.0)
        assert player.position.current_sample == 0

    def test_seek_no_audio_noop(self) -> None:
        player = AudioPlayer()
        player.seek(1.0)  # should not raise
        assert player.position.current_sample == 0


class TestAudioCallback:
    def _outdata(self, frames, channels=1):
        return np.zeros((frames, channels), dtype=np.float32)

    def test_normal_chunk_mono(self, mock_sd, mono_audio) -> None:
        player = AudioPlayer()
        player.load(mono_audio, 16000)
        player._state = PlaybackState.PLAYING
        out = self._outdata(100)
        player._audio_callback(out, 100, None, None)
        assert player._position == 100
        assert np.allclose(out[:, 0], mono_audio[:100])

    def test_callback_not_playing_fills_zero(self, mock_sd, mono_audio) -> None:
        player = AudioPlayer()
        player.load(mono_audio, 16000)  # state STOPPED
        out = self._outdata(50)
        out.fill(1.0)
        player._audio_callback(out, 50, None, None)
        assert np.all(out == 0)

    def test_callback_end_no_loop_raises_stop(self, mock_sd, mono_audio) -> None:
        fake, _ = mock_sd
        player = AudioPlayer()
        player.load(mono_audio, 16000)
        player._state = PlaybackState.PLAYING
        player._position = len(mono_audio) - 10
        out = self._outdata(100)
        with pytest.raises(fake.CallbackStop):
            player._audio_callback(out, 100, None, None)
        assert player._position == len(mono_audio)

    def test_callback_past_end_no_loop_raises_stop(self, mock_sd, mono_audio) -> None:
        fake, _ = mock_sd
        player = AudioPlayer()
        player.load(mono_audio, 16000)
        player._state = PlaybackState.PLAYING
        player._position = len(mono_audio)
        out = self._outdata(100)
        with pytest.raises(fake.CallbackStop):
            player._audio_callback(out, 100, None, None)

    def test_callback_loop_wraps(self, mock_sd, mono_audio) -> None:
        player = AudioPlayer()
        player.load(mono_audio, 16000)
        player._state = PlaybackState.PLAYING
        player._loop = True
        player._position = len(mono_audio) - 10
        out = self._outdata(100)
        player._audio_callback(out, 100, None, None)
        assert player._position == 90  # wrapped

    def test_callback_loop_restart_from_end(self, mock_sd, mono_audio) -> None:
        player = AudioPlayer()
        player.load(mono_audio, 16000)
        player._state = PlaybackState.PLAYING
        player._loop = True
        player._position = len(mono_audio)
        out = self._outdata(100)
        player._audio_callback(out, 100, None, None)
        assert player._position == 100

    def test_callback_position_change_invoked(self, mock_sd, mono_audio) -> None:
        calls = []
        player = AudioPlayer()
        player.load(mono_audio, 16000, on_position_change=calls.append)
        player._state = PlaybackState.PLAYING
        player._audio_callback(self._outdata(100), 100, None, None)
        assert len(calls) == 1

    def test_callback_position_change_error_swallowed(self, mock_sd, mono_audio) -> None:
        def boom(_pos):
            raise RuntimeError("cb fail")

        player = AudioPlayer()
        player.load(mono_audio, 16000, on_position_change=boom)
        player._state = PlaybackState.PLAYING
        # Should not raise despite callback error
        player._audio_callback(self._outdata(100), 100, None, None)

    def test_callback_with_status_logs(self, mock_sd, mono_audio) -> None:
        player = AudioPlayer()
        player.load(mono_audio, 16000)
        player._state = PlaybackState.PLAYING
        player._audio_callback(self._outdata(100), 100, None, "underflow")

    def test_callback_stereo_normal(self, mock_sd) -> None:
        stereo = np.zeros((16000, 2), dtype=np.float32)
        player = AudioPlayer()
        player.load(stereo, 16000)
        player._state = PlaybackState.PLAYING
        out = self._outdata(100, channels=2)
        player._audio_callback(out, 100, None, None)
        assert player._position == 100


class TestFinishedCallback:
    def test_finished_resets_and_calls(self) -> None:
        called = []
        player = AudioPlayer()
        player.load(np.zeros(100, dtype=np.float32), 16000, on_complete=lambda: called.append(1))
        player._state = PlaybackState.PLAYING
        player._position = 50
        player._finished_callback()
        assert player.is_stopped
        assert player._position == 0
        assert called == [1]

    def test_finished_loop_no_reset(self) -> None:
        player = AudioPlayer()
        player.load(np.zeros(100, dtype=np.float32), 16000)
        player._loop = True
        player._state = PlaybackState.PLAYING
        player._finished_callback()
        assert player.state == PlaybackState.PLAYING

    def test_finished_callback_error_swallowed(self) -> None:
        def boom():
            raise RuntimeError("done fail")

        player = AudioPlayer()
        player.load(np.zeros(100, dtype=np.float32), 16000, on_complete=boom)
        player._finished_callback()  # should not raise


class TestLoadFile:
    def test_load_file(self, mono_audio) -> None:
        player = AudioPlayer()
        with patch("bioamla.audio._pydub.load_audio", return_value=(mono_audio, 16000)):
            player.load_file("/fake.wav")
        assert player.position.total_samples == len(mono_audio)


class TestConvenienceFunctions:
    def test_play_audio_array(self, mock_sd, mono_audio) -> None:
        playback_mod._global_player = None
        player = play_audio(mono_audio, sample_rate=16000)
        assert player.is_playing
        stop_audio()
        assert player.is_stopped

    def test_play_audio_array_requires_sr(self, mock_sd, mono_audio) -> None:
        playback_mod._global_player = None
        with pytest.raises(InvalidInputError):
            play_audio(mono_audio)

    def test_play_audio_reuses_global(self, mock_sd, mono_audio) -> None:
        playback_mod._global_player = None
        p1 = play_audio(mono_audio, sample_rate=16000)
        p2 = play_audio(mono_audio, sample_rate=16000)
        assert p1 is p2

    def test_play_audio_filepath(self, mock_sd, mono_audio) -> None:
        playback_mod._global_player = None
        with patch("bioamla.audio._pydub.load_audio", return_value=(mono_audio, 16000)):
            player = play_audio("/fake.wav")
        assert player.is_playing

    def test_play_audio_block(self, mock_sd, mono_audio) -> None:
        fake, _ = mock_sd
        playback_mod._global_player = None
        # Make is_playing become False after first sleep call.
        with patch.object(AudioPlayer, "is_playing", new=False):
            play_audio(mono_audio, sample_rate=16000, block=True)

    def test_stop_audio_no_global(self) -> None:
        playback_mod._global_player = None
        stop_audio()  # should not raise
