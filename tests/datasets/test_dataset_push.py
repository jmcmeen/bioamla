"""Tests for the dataset card builder, write_dataset_card, and `catalogs hf push-dataset`."""

from click.testing import CliRunner

import bioamla.catalogs.huggingface as hf
from bioamla.catalogs._models import PushResult
from bioamla.cli.commands.catalogs import catalogs
from bioamla.datasets import (
    DatasetManifest,
    build_dataset_card,
    create_label_map,
    write_dataset_card,
)
from bioamla.datasets._metadata import write_metadata_csv


def _manifest():
    return DatasetManifest(
        name="demo",
        kind="partitioned",
        label2id=create_label_map(["call", "chorus"]),
        id2label={0: "call", 1: "chorus"},
        class_counts={"call": 8, "chorus": 4},
        splits={"train": 8, "test": 4},
        sources=[{"source": "xeno_canto", "files": 12}],
        sample_rate=16000,
    )


def _write_dataset(tmp_path):
    rows = [
        {"file_name": "call/a.wav", "label": "call", "split": "train"},
        {"file_name": "chorus/b.wav", "label": "chorus", "split": "test"},
    ]
    write_metadata_csv(tmp_path / "metadata.csv", rows, merge_existing=False)
    return tmp_path


class TestBuildDatasetCard:
    def test_has_frontmatter_and_class_table(self):
        card = build_dataset_card(_manifest())
        assert card.startswith("---\n")
        assert "task_categories:" in card
        assert "- audio-classification" in card
        assert "# demo" in card
        assert "| call | 0 | 8 |" in card
        assert "| chorus | 1 | 4 |" in card
        assert "xeno_canto: 12 files" in card

    def test_empty_manifest_still_renders(self):
        card = build_dataset_card(DatasetManifest(name="empty"))
        assert "# empty" in card
        assert "Total clips/files:** 0" in card


class TestWriteDatasetCard:
    def test_builds_from_metadata(self, tmp_path):
        d = _write_dataset(tmp_path)
        path = write_dataset_card(str(d))
        assert path == str(d / "README.md")
        assert "task_categories:" in (d / "README.md").read_text()

    def test_returns_none_when_not_a_dataset(self, tmp_path):
        # No metadata.csv and no dataset.json -> nothing to build a card from.
        assert write_dataset_card(str(tmp_path)) is None
        assert not (tmp_path / "README.md").exists()


class TestPushDatasetCli:
    def test_writes_card_and_delegates(self, tmp_path, monkeypatch):
        d = _write_dataset(tmp_path)
        calls = {}

        def fake_push(path, repo_id, private=False, commit_message=None, **kwargs):
            calls.update(path=path, repo_id=repo_id, private=private, **kwargs)
            return PushResult(
                repo_id=repo_id,
                repo_type="dataset",
                url=f"https://huggingface.co/datasets/{repo_id}",
                files_uploaded=3,
                total_size_bytes=123,
            )

        monkeypatch.setattr(hf, "push_dataset", fake_push)

        res = CliRunner().invoke(catalogs, ["hf", "push-dataset", str(d), "me/frogs", "--private"])
        assert res.exit_code == 0, res.output
        assert (d / "README.md").exists()
        assert "task_categories:" in (d / "README.md").read_text()
        assert calls["repo_id"] == "me/frogs"
        assert calls["private"] is True
        assert "huggingface.co/datasets/me/frogs" in res.output

    def test_no_card_flag_skips_readme(self, tmp_path, monkeypatch):
        d = _write_dataset(tmp_path)
        monkeypatch.setattr(
            hf, "push_dataset", lambda *a, **k: PushResult("r", "dataset", "u", 1, 1)
        )
        res = CliRunner().invoke(catalogs, ["hf", "push-dataset", str(d), "me/frogs", "--no-card"])
        assert res.exit_code == 0, res.output
        assert not (d / "README.md").exists()

    def test_non_dataset_folder_pushes_without_card(self, tmp_path, monkeypatch):
        # A folder with no manifest/metadata is still pushable; just no card.
        (tmp_path / "audio.wav").write_bytes(b"RIFF")
        pushed = {"n": 0}

        def fake_push(*a, **k):
            pushed["n"] += 1
            return PushResult("r", "dataset", "u", 1, 1)

        monkeypatch.setattr(hf, "push_dataset", fake_push)
        res = CliRunner().invoke(catalogs, ["hf", "push-dataset", str(tmp_path), "me/x"])
        assert res.exit_code == 0, res.output
        assert pushed["n"] == 1
        assert not (tmp_path / "README.md").exists()

    def test_push_failure_surfaces_login_hint(self, tmp_path, monkeypatch):
        from bioamla.exceptions import CatalogError

        d = _write_dataset(tmp_path)

        def boom(*a, **k):
            raise CatalogError("Push failed: 401")

        monkeypatch.setattr(hf, "push_dataset", boom)
        res = CliRunner().invoke(catalogs, ["hf", "push-dataset", str(d), "me/frogs"])
        assert res.exit_code != 0
        assert "huggingface-cli login" in res.output


class TestDatasetLicense:
    """The dataset's license in its card, and the licenses its files carry."""

    def _dataset(self, tmp_path):
        rows = [
            {"file_name": "a.wav", "label": "call", "license": "CC BY 4.0"},
            {"file_name": "b.wav", "label": "call", "license": "CC BY 4.0", "attr_lic": "CC0"},
            {"file_name": "c.wav", "label": "chorus", "license": "CC BY-NC 4.0"},
        ]
        write_metadata_csv(tmp_path / "metadata.csv", rows, merge_existing=False)
        return tmp_path

    def test_manifest_counts_file_licenses_attr_lic_first(self, tmp_path):
        from bioamla.datasets import build_manifest_from_metadata

        manifest = build_manifest_from_metadata(str(self._dataset(tmp_path)))
        assert manifest.licenses == {"CC BY 4.0": 1, "CC BY-NC 4.0": 1, "CC0": 1}

    def test_card_front_matter_and_section(self):
        manifest = _manifest()
        manifest.licenses = {"CC BY-NC 4.0": 12}
        card = build_dataset_card(manifest, license="CC-BY-NC-4.0")
        front = card.split("---")[1]
        assert "license: cc-by-nc-4.0" in front
        assert "- **Dataset:** `cc-by-nc-4.0`" in card
        assert "- **Files:** CC BY-NC 4.0 (12)" in card

    def test_no_license_no_front_matter_line(self):
        assert "license:" not in build_dataset_card(_manifest())

    def test_a_bad_license_id_is_refused(self):
        import pytest

        from bioamla.exceptions import InvalidInputError

        for bad in ("CC BY NC", "cc-by-nc", "cc-by-sa-2.0"):
            with pytest.raises(InvalidInputError, match="Hub license id"):
                build_dataset_card(_manifest(), license=bad)

    def test_a_quoted_card_license_is_read_whole(self):
        from bioamla.datasets.manifest import card_license

        assert card_license('---\nlicense: "cc-by-4.0"\ntags:\n- audio\n---\n') == "cc-by-4.0"
        assert card_license("---\ntags:\n- audio\n---\nlicense: mit\n") is None

    def test_a_license_without_anything_to_build_a_card_from_stops_the_push(self, tmp_path):
        res = CliRunner().invoke(
            catalogs, ["hf", "push-dataset", str(tmp_path), "me/frogs", "--license", "cc0-1.0"]
        )
        assert res.exit_code != 0
        assert "no metadata.csv" in res.output

    def test_a_redacted_push_keeps_the_cards_license(self, tmp_path):
        from bioamla.datasets import redacted_dataset_files

        d = self._dataset(tmp_path)
        write_dataset_card(str(d), license="cc-by-nc-4.0")
        card = redacted_dataset_files(d, "strip")["README.md"].decode()
        assert "license: cc-by-nc-4.0" in card.split("---")[1]

    def test_cli_writes_the_license_to_the_card(self, tmp_path, monkeypatch):
        d = self._dataset(tmp_path)
        monkeypatch.setattr(
            hf,
            "push_dataset",
            lambda path, repo_id, **kw: PushResult(
                repo_id=repo_id, repo_type="dataset", url="u", files_uploaded=1,
                total_size_bytes=1,
            ),
        )  # fmt: skip
        res = CliRunner().invoke(
            catalogs, ["hf", "push-dataset", str(d), "me/frogs", "--license", "CC-BY-NC-4.0"]
        )
        assert res.exit_code == 0, res.output
        assert "license: cc-by-nc-4.0" in (d / "README.md").read_text()

    def test_license_needs_the_card(self, tmp_path):
        res = CliRunner().invoke(
            catalogs,
            ["hf", "push-dataset", str(self._dataset(tmp_path)), "me/frogs", "--no-card",
             "--license", "cc0-1.0"],
        )  # fmt: skip
        assert res.exit_code != 0
        assert "--no-card" in res.output
