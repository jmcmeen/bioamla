"""Tests for the dataset card builder and `dataset push` CLI (no network)."""

from click.testing import CliRunner

import bioamla.catalogs.huggingface as hf
from bioamla.catalogs._models import PushResult
from bioamla.cli.commands.dataset import dataset
from bioamla.datasets import DatasetManifest, build_dataset_card, create_label_map
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


def _write_dataset(tmp_path):
    rows = [
        {"file_name": "call/a.wav", "label": "call", "split": "train"},
        {"file_name": "chorus/b.wav", "label": "chorus", "split": "test"},
    ]
    write_metadata_csv(tmp_path / "metadata.csv", rows, merge_existing=False)
    return tmp_path


class TestDatasetPushCli:
    def test_writes_card_and_delegates(self, tmp_path, monkeypatch):
        d = _write_dataset(tmp_path)
        calls = {}

        def fake_push(path, repo_id, private=False, commit_message=None):
            calls.update(path=path, repo_id=repo_id, private=private, commit_message=commit_message)
            return PushResult(
                repo_id=repo_id,
                repo_type="dataset",
                url=f"https://huggingface.co/datasets/{repo_id}",
                files_uploaded=3,
                total_size_bytes=123,
            )

        monkeypatch.setattr(hf, "push_dataset", fake_push)

        res = CliRunner().invoke(dataset, ["push", str(d), "me/frogs", "--private"])
        assert res.exit_code == 0, res.output
        assert (d / "README.md").exists()
        assert "task_categories:" in (d / "README.md").read_text()
        assert calls["repo_id"] == "me/frogs"
        assert calls["private"] is True
        assert "huggingface.co/datasets/me/frogs" in res.output

    def test_no_card_flag_skips_readme(self, tmp_path, monkeypatch):
        d = _write_dataset(tmp_path)
        monkeypatch.setattr(
            hf,
            "push_dataset",
            lambda *a, **k: PushResult("r", "dataset", "u", 1, 1),
        )
        res = CliRunner().invoke(dataset, ["push", str(d), "me/frogs", "--no-card"])
        assert res.exit_code == 0, res.output
        assert not (d / "README.md").exists()

    def test_rejects_non_dataset_dir(self, tmp_path, monkeypatch):
        # An empty directory is not a dataset; should error before any push.
        called = {"n": 0}

        def fake_push(*a, **k):
            called["n"] += 1
            return PushResult("r", "dataset", "u", 1, 1)

        monkeypatch.setattr(hf, "push_dataset", fake_push)
        res = CliRunner().invoke(dataset, ["push", str(tmp_path), "me/frogs"])
        assert res.exit_code != 0
        assert "doesn't look like a dataset" in res.output
        assert called["n"] == 0

    def test_push_failure_surfaces_login_hint(self, tmp_path, monkeypatch):
        from bioamla.exceptions import CatalogError

        d = _write_dataset(tmp_path)

        def boom(*a, **k):
            raise CatalogError("Push failed: 401")

        monkeypatch.setattr(hf, "push_dataset", boom)
        res = CliRunner().invoke(dataset, ["push", str(d), "me/frogs"])
        assert res.exit_code != 0
        assert "huggingface-cli login" in res.output
