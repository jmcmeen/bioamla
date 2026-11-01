"""Tests for the restored ``batch audio convert`` capability."""

from pathlib import Path

import pytest

from bioamla.audio.convert import convert_audio_file


class TestConvertAudioFile:
    @pytest.mark.usefixtures("requires_ffmpeg_cli")
    def test_converts_format(self, test_audio_path, tmp_path):
        out = tmp_path / "out.flac"
        result = convert_audio_file(test_audio_path, out, target_format="flac")
        assert Path(result).exists()
        assert Path(result).suffix == ".flac"

    def test_resamples(self, test_audio_path, tmp_path):
        from bioamla.audio.info import get_audio_info

        out = tmp_path / "out.wav"
        convert_audio_file(test_audio_path, out, target_format="wav", target_sample_rate=8000)
        assert get_audio_info(str(out)).sample_rate == 8000

    def test_rechannel_to_stereo(self, test_audio_path, tmp_path):
        from bioamla.audio.info import get_audio_info

        out = tmp_path / "stereo.wav"
        convert_audio_file(test_audio_path, out, target_format="wav", target_channels=2)
        assert get_audio_info(str(out)).channels == 2

    @pytest.mark.usefixtures("requires_ffmpeg_cli")
    def test_delete_original(self, test_audio_path, tmp_path):
        # Copy fixture into tmp so deletion is safe.
        src = tmp_path / "src.wav"
        src.write_bytes(Path(test_audio_path).read_bytes())
        out = tmp_path / "out.flac"
        convert_audio_file(src, out, target_format="flac", delete_original=True)
        assert not src.exists()
        assert out.exists()


class TestConvertKeepsChannels:
    @pytest.fixture
    def stereo(self, tmp_path):
        import numpy as np
        import soundfile as sf

        t = np.arange(8000) / 8000
        left, right = 0.5 * np.sin(2 * np.pi * 440 * t), 0.25 * np.sin(2 * np.pi * 880 * t)
        path = tmp_path / "stereo.wav"
        sf.write(path, np.column_stack([left, right]), 8000)
        return path

    def test_stereo_stays_stereo_with_its_channels_apart(self, stereo, tmp_path):
        import numpy as np
        import soundfile as sf

        out = tmp_path / "out.wav"
        convert_audio_file(stereo, out, target_format="wav")
        data, sr = sf.read(out)
        original, _ = sf.read(stereo)
        assert data.shape == original.shape and sr == 8000
        assert np.allclose(data, original, atol=1e-3)  # not a downmix duplicated

    def test_resampling_keeps_the_channels(self, stereo, tmp_path):
        import soundfile as sf

        out = tmp_path / "out.wav"
        convert_audio_file(stereo, out, target_format="wav", target_sample_rate=4000)
        data, sr = sf.read(out)
        assert sr == 4000 and data.shape == (4000, 2)

    def test_mono_on_request(self, stereo, tmp_path):
        import soundfile as sf

        out = tmp_path / "out.wav"
        convert_audio_file(stereo, out, target_format="wav", target_channels=1)
        assert sf.info(out).channels == 1

    @pytest.mark.usefixtures("requires_ffmpeg_cli")
    def test_other_formats_keep_their_channels_too(self, stereo, tmp_path):
        import soundfile as sf

        mp3 = tmp_path / "s.mp3"
        convert_audio_file(stereo, mp3, target_format="mp3")
        out = tmp_path / "back.wav"
        convert_audio_file(mp3, out, target_format="wav")
        assert sf.info(out).channels == 2

    def test_load_audio_channels(self, stereo, test_audio_path):
        from bioamla.audio import load_audio_channels

        audio, sr = load_audio_channels(stereo)
        assert audio.shape == (8000, 2) and audio.dtype.name == "float32"
        mono, _ = load_audio_channels(test_audio_path)
        assert mono.ndim == 1


class TestConvertCli:
    @pytest.mark.usefixtures("requires_ffmpeg_cli")
    def test_directory_mode_produces_target_format(self, test_audio_dir, tmp_path):
        from click.testing import CliRunner

        from bioamla.cli.commands.batch import batch

        out_dir = tmp_path / "converted"
        runner = CliRunner()
        res = runner.invoke(
            batch,
            [
                "audio",
                "convert",
                "--input-dir",
                test_audio_dir,
                "--output-dir",
                str(out_dir),
                "--format",
                "flac",
                "--quiet",
            ],
        )
        assert res.exit_code == 0, res.output
        flac_files = list(out_dir.glob("*.flac"))
        assert len(flac_files) == 3

    @pytest.mark.usefixtures("requires_ffmpeg_cli")
    def test_csv_mode_updates_paths(self, test_audio_dir, tmp_path):
        import csv

        from click.testing import CliRunner

        from bioamla.cli.commands.batch import batch

        csv_dir = Path(test_audio_dir)
        csv_path = csv_dir / "meta.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file_name"])
            w.writeheader()
            w.writerow({"file_name": "audio_0.wav"})

        out_dir = tmp_path / "out"
        runner = CliRunner()
        res = runner.invoke(
            batch,
            [
                "audio",
                "convert",
                "--input-file",
                str(csv_path),
                "--output-dir",
                str(out_dir),
                "--format",
                "flac",
                "--quiet",
            ],
        )
        assert res.exit_code == 0, res.output
        out_csv = out_dir / "meta.csv"
        with out_csv.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # Path updated to converted .flac file, relative to output_dir.
        assert rows[0]["file_name"] == "audio_0.flac"
        assert (out_dir / "audio_0.flac").exists()
