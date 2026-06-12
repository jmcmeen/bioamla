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
