"""CLI tests for license traceability: extract-clips --source-metadata, license --format md, build."""

import csv

import numpy as np
from click.testing import CliRunner

from bioamla.audio import save_audio
from bioamla.cli.commands.dataset import dataset
from bioamla.datasets import Annotation, save_bioamla_annotations
from bioamla.datasets._metadata import write_metadata_csv


def _catalog_dir(tmp_path, *, license="CC-BY-NC", with_metadata=True):
    """A fake catalog download dir: rec.wav + annotations (+ metadata.csv)."""
    d = tmp_path / "cat"
    d.mkdir()
    sr = 16000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False, dtype=np.float32)
    save_audio(str(d / "rec.wav"), 0.5 * np.sin(2 * np.pi * 440 * t), sr)
    save_bioamla_annotations(
        [
            Annotation(start_time=0.1, end_time=0.5, label="call"),
            Annotation(start_time=1.0, end_time=1.4, label="chorus"),
        ],
        str(d / "rec.json"),
    )
    if with_metadata:
        write_metadata_csv(
            d / "metadata.csv",
            [
                {
                    "file_name": "rec.wav",
                    "source": "xeno_canto",
                    "license": license,
                    "attribution": "J. Doe",
                    "attr_url": "https://xeno-canto.org/123",
                }
            ],
            merge_existing=False,
        )
    return d


class TestExtractClipsSourceMetadata:
    def test_auto_detects_sibling_metadata(self, tmp_path):
        d = _catalog_dir(tmp_path)
        out = tmp_path / "ds"
        res = CliRunner().invoke(dataset, ["extract-clips", str(d), str(out), "--layout", "flat"])
        assert res.exit_code == 0, res.output
        assert "Provenance: joined 2/2" in res.output
        with (out / "metadata.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert all(r["license"] == "CC-BY-NC" for r in rows)

    def test_explicit_source_metadata_flag(self, tmp_path):
        d = _catalog_dir(tmp_path, with_metadata=False)
        meta = tmp_path / "m.csv"
        write_metadata_csv(
            meta,
            [{"file_name": "rec.wav", "license": "CC0", "attribution": "A"}],
            merge_existing=False,
        )
        out = tmp_path / "ds"
        res = CliRunner().invoke(
            dataset,
            ["extract-clips", str(d), str(out), "--layout", "flat", "--source-metadata", str(meta)],
        )
        assert res.exit_code == 0, res.output
        with (out / "metadata.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert all(r["license"] == "CC0" for r in rows)


class TestLicenseFormatMd:
    def test_generates_markdown(self, tmp_path):
        d = tmp_path / "ds"
        d.mkdir()
        write_metadata_csv(
            d / "metadata.csv",
            [{"file_name": "a.wav", "source": "xeno_canto", "license": "CC-BY", "attr_url": "u"}],
            merge_existing=False,
        )
        res = CliRunner().invoke(dataset, ["license", str(d), "--format", "md", "--quiet"])
        assert res.exit_code == 0, res.output
        assert (d / "ATTRIBUTIONS.md").exists()
        assert "## xeno_canto" in (d / "ATTRIBUTIONS.md").read_text()


class TestBuildAttributions:
    def test_build_writes_attributions_md(self, tmp_path):
        d = _catalog_dir(tmp_path)
        out = tmp_path / "ds"
        res = CliRunner().invoke(dataset, ["build", str(d), str(out), "--name", "demo"])
        assert res.exit_code == 0, res.output
        assert "attributions:" in res.output
        md = (out / "ATTRIBUTIONS.md").read_text()
        assert "CC-BY-NC" in md and "xeno-canto.org/123" in md

    def test_no_attributions_flag_skips(self, tmp_path):
        d = _catalog_dir(tmp_path)
        out = tmp_path / "ds"
        res = CliRunner().invoke(dataset, ["build", str(d), str(out), "--no-attributions"])
        assert res.exit_code == 0, res.output
        assert not (out / "ATTRIBUTIONS.md").exists()

    def test_build_without_provenance_skips_silently(self, tmp_path):
        # No catalog metadata.csv -> no provenance -> build succeeds, no ATTRIBUTIONS.md.
        d = _catalog_dir(tmp_path, with_metadata=False)
        out = tmp_path / "ds"
        res = CliRunner().invoke(dataset, ["build", str(d), str(out)])
        assert res.exit_code == 0, res.output
        assert not (out / "ATTRIBUTIONS.md").exists()
        assert "attributions:" not in res.output
