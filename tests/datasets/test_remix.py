"""Tests for :mod:`bioamla.datasets.remix`."""

from __future__ import annotations

import csv

import pytest

from bioamla.datasets._metadata import read_metadata_csv, write_metadata_csv
from bioamla.datasets.remix import RemixRecipe, RemixSource, remix_dataset
from bioamla.exceptions import InvalidInputError, NotFoundError, RemixError


def _make_dataset(root, rows: list[dict]) -> None:
    """Create a dataset dir with metadata.csv and the referenced audio files."""
    root.mkdir(parents=True, exist_ok=True)
    fieldnames: set[str] = set()
    for r in rows:
        fieldnames.update(r.keys())
    write_metadata_csv(root / "metadata.csv", rows, fieldnames, merge_existing=False)
    for r in rows:
        path = root / r["file_name"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFFfake")


def _esc_rows(*labels: str, per_label: int = 2) -> list[dict]:
    """ESC-50 style rows: ``category`` label column plus attribution columns."""
    rows = []
    for t, label in enumerate(labels):
        for i in range(per_label):
            rows.append(
                {
                    "file_name": f"{label}/{t}-{i}.wav",
                    "category": label,
                    "target": str(t),
                    "fold": "1",
                    "attr_id": f"user{t}",
                    "attr_lic": "CC0",
                    "attr_url": f"https://example.org/{t}",
                    "attr_note": "",
                }
            )
    return rows


@pytest.fixture
def esc(tmp_path):
    root = tmp_path / "esc"
    _make_dataset(root, _esc_rows("dog", "cat", "airplane"))
    return root


def _labels(output) -> dict[str, int]:
    rows, _ = read_metadata_csv(output / "metadata.csv")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    return counts


class TestRecipe:
    def test_keep_and_drop_exclusive(self) -> None:
        with pytest.raises(InvalidInputError, match="keep or drop"):
            RemixRecipe(source="x", keep=["a"], drop=["b"])

    def test_source_required(self) -> None:
        with pytest.raises(InvalidInputError, match="source"):
            RemixRecipe(source="")

    def test_duplicate_add_replace_label(self) -> None:
        with pytest.raises(InvalidInputError, match="more than once"):
            RemixRecipe(source="x", add=[RemixSource("a", "p")], replace=[RemixSource("a", "q")])

    def test_from_dict_full(self) -> None:
        recipe = RemixRecipe.from_dict(
            {
                "remix": {"name": "n", "source": "src", "drop": ["x"]},
                "add": [{"label": "a", "source": "pa", "max_clips": 3}],
                "replace": [{"label": "b", "source": "pb"}],
            }
        )
        assert recipe.name == "n"
        assert recipe.drop == ["x"] and recipe.keep is None
        assert recipe.add == [RemixSource("a", "pa", 3)]
        assert recipe.replace == [RemixSource("b", "pb", None)]

    @pytest.mark.parametrize(
        "data, match",
        [
            ({}, r"\[remix\] table"),
            ({"remix": {"source": "s"}, "bogus": {}}, "Unknown recipe section"),
            ({"remix": {"source": "s", "colour": 1}}, r"Unknown \[remix\] key"),
            ({"remix": {"source": "s"}, "add": [{"label": "a"}]}, "must be a non-empty path"),
            ({"remix": {"source": "s"}, "add": [{"label": "a", "source": "p", "x": 1}]}, "Unknown"),
            ({"remix": {"source": "s", "keep": "dog"}}, "list of label strings"),
            ({"remix": {"source": "s", "drop": [1]}}, "list of label strings"),
            ({"remix": {"source": 3}}, "non-empty path"),
            (
                {
                    "remix": {"source": "s"},
                    "add": [{"label": "a", "source": "p", "max_clips": "3"}],
                },
                "max_clips",
            ),
            (
                {
                    "remix": {"source": "s"},
                    "add": [{"label": "a", "source": "p", "max_clips": True}],
                },
                "max_clips",
            ),
            ({"remix": {"source": "s"}, "add": {"label": "a", "source": "p"}}, "array of tables"),
        ],
    )
    def test_from_dict_rejects_malformed(self, data, match) -> None:
        with pytest.raises(InvalidInputError, match=match):
            RemixRecipe.from_dict(data)

    def test_from_toml(self, tmp_path) -> None:
        path = tmp_path / "r.toml"
        path.write_text(
            '[remix]\nsource = "s"\nkeep = ["dog"]\n\n[[add]]\nlabel = "x"\nsource = "p"\n'
        )
        recipe = RemixRecipe.from_toml(path)
        assert recipe.keep == ["dog"]
        assert recipe.add[0].label == "x"

    def test_from_toml_missing(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            RemixRecipe.from_toml(tmp_path / "nope.toml")


class TestRemixDataset:
    def test_keep(self, esc, tmp_path) -> None:
        out = tmp_path / "out"
        stats = remix_dataset(RemixRecipe(source=str(esc), keep=["dog", "cat"]), out)
        assert _labels(out) == {"dog": 2, "cat": 2}
        assert stats["labels"] == ["cat", "dog"]
        assert stats["dropped"] == 2
        assert stats["files_copied"] == 4
        assert stats["label_column"] == "category"
        assert (out / "dog" / "0-0.wav").read_bytes() == b"RIFFfake"
        assert (out / "LICENSE").exists()
        assert stats["license_file"] == str(out / "LICENSE")

    def test_drop(self, esc, tmp_path) -> None:
        out = tmp_path / "out"
        stats = remix_dataset(RemixRecipe(source=str(esc), drop=["airplane"]), out)
        assert _labels(out) == {"dog": 2, "cat": 2}
        assert stats["dropped"] == 2
        assert not (out / "airplane").exists()

    def test_no_filter_copies_everything(self, esc, tmp_path) -> None:
        stats = remix_dataset(RemixRecipe(source=str(esc)), tmp_path / "out")
        assert stats["total_files"] == 6 and stats["dropped"] == 0

    def test_passthrough_columns(self, esc, tmp_path) -> None:
        out = tmp_path / "out"
        remix_dataset(RemixRecipe(source=str(esc), keep=["cat"]), out)
        rows, fields = read_metadata_csv(out / "metadata.csv")
        assert {"target", "fold", "attr_url"} <= fields
        assert rows[0]["target"] == "1"

    def test_unknown_label_rejected(self, esc, tmp_path) -> None:
        with pytest.raises(InvalidInputError, match="not in the source: horse"):
            remix_dataset(RemixRecipe(source=str(esc), keep=["dog", "horse"]), tmp_path / "o")
        with pytest.raises(InvalidInputError, match="Cannot drop"):
            remix_dataset(RemixRecipe(source=str(esc), drop=["horse"]), tmp_path / "o")
        assert not (tmp_path / "o").exists()

    def test_add_class(self, esc, tmp_path) -> None:
        extra = tmp_path / "frogs"
        _make_dataset(
            extra,
            [
                {"file_name": f"clip{i}.wav", "label": "Pseudacris crucifer", "attr_lic": "cc-by"}
                for i in range(5)
            ],
        )
        out = tmp_path / "out"
        recipe = RemixRecipe(
            source=str(esc), keep=["dog"], add=[RemixSource("spring_peeper", str(extra), 3)]
        )
        stats = remix_dataset(recipe, out)
        assert _labels(out) == {"dog": 2, "spring_peeper": 3}
        assert stats["added"] == 1
        assert sorted(p.name for p in (out / "spring_peeper").iterdir()) == [
            "clip0.wav",
            "clip1.wav",
            "clip2.wav",
        ]
        rows, _ = read_metadata_csv(out / "metadata.csv")
        peeper = [r for r in rows if r["category"] == "spring_peeper"]
        # Every label-like column carries the remix label, so tools that prefer
        # ``label`` over ``category`` agree with the remix.
        assert peeper[0]["label"] == "spring_peeper"
        assert peeper[0]["attr_lic"] == "cc-by"

    def test_add_existing_label_rejected(self, esc, tmp_path) -> None:
        extra = tmp_path / "more"
        _make_dataset(extra, [{"file_name": "x.wav", "category": "dog"}])
        with pytest.raises(InvalidInputError, match="use \\[\\[replace\\]\\]"):
            remix_dataset(
                RemixRecipe(source=str(esc), add=[RemixSource("dog", str(extra))]),
                tmp_path / "out",
            )

    def test_replace_class(self, esc, tmp_path) -> None:
        extra = tmp_path / "newdogs"
        _make_dataset(extra, [{"file_name": f"d{i}.wav", "category": "dog"} for i in range(3)])
        out = tmp_path / "out"
        stats = remix_dataset(
            RemixRecipe(source=str(esc), replace=[RemixSource("dog", str(extra))]), out
        )
        assert _labels(out) == {"dog": 3, "cat": 2, "airplane": 2}
        assert stats["replaced"] == 1
        assert not (out / "dog" / "0-0.wav").exists()
        assert (out / "dog" / "d1.wav").exists()

    def test_replace_unknown_label_rejected(self, esc, tmp_path) -> None:
        extra = tmp_path / "x"
        _make_dataset(extra, [{"file_name": "x.wav", "category": "horse"}])
        with pytest.raises(InvalidInputError, match="Cannot replace"):
            remix_dataset(
                RemixRecipe(source=str(esc), replace=[RemixSource("horse", str(extra))]),
                tmp_path / "out",
            )

    def test_explicit_label_column(self, tmp_path) -> None:
        src = tmp_path / "src"
        _make_dataset(
            src,
            [
                {"file_name": "a.wav", "label": "x", "group": "g1"},
                {"file_name": "b.wav", "label": "y", "group": "g2"},
            ],
        )
        out = tmp_path / "out"
        stats = remix_dataset(RemixRecipe(source=str(src), label_column="group", keep=["g2"]), out)
        assert stats["labels"] == ["g2"]
        assert (out / "g2" / "b.wav").exists()
        with pytest.raises(InvalidInputError, match="Label column"):
            remix_dataset(RemixRecipe(source=str(src), label_column="nope"), tmp_path / "o2")

    def test_no_label_column(self, tmp_path) -> None:
        src = tmp_path / "src"
        _make_dataset(src, [{"file_name": "a.wav", "other": "x"}])
        with pytest.raises(InvalidInputError, match="neither"):
            remix_dataset(RemixRecipe(source=str(src)), tmp_path / "out")

    def test_missing_source_file_skipped(self, esc, tmp_path) -> None:
        (esc / "dog" / "0-0.wav").unlink()
        stats = remix_dataset(RemixRecipe(source=str(esc), keep=["dog"]), tmp_path / "out")
        assert stats["files_copied"] == 1 and stats["files_missing"] == 1
        assert stats["total_files"] == 1

    def test_all_source_files_missing_raises(self, esc, tmp_path) -> None:
        for p in (esc / "dog").iterdir():
            p.unlink()
        with pytest.raises(RemixError, match="No clips could be copied"):
            remix_dataset(RemixRecipe(source=str(esc), keep=["dog"]), tmp_path / "out")
        assert not (tmp_path / "out" / "metadata.csv").exists()

    def test_duplicate_basenames_counted(self, esc, tmp_path) -> None:
        src = tmp_path / "src"
        _make_dataset(
            src,
            [
                {"file_name": "a/x.wav", "label": "dog", "attr_lic": "CC0"},
                {"file_name": "b/x.wav", "label": "dog", "attr_lic": "CC0"},
            ],
        )
        stats = remix_dataset(RemixRecipe(source=str(src)), tmp_path / "out")
        assert stats["files_copied"] == 1 and stats["files_duplicate"] == 1

    def test_missing_source_dir(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            remix_dataset(RemixRecipe(source=str(tmp_path / "nope")), tmp_path / "out")

    def test_output_already_a_dataset(self, esc, tmp_path) -> None:
        with pytest.raises(InvalidInputError, match="already contains"):
            remix_dataset(RemixRecipe(source=str(esc)), esc)

    def test_no_attribution_skips_license(self, tmp_path) -> None:
        src = tmp_path / "src"
        _make_dataset(src, [{"file_name": "a.wav", "label": "x"}])
        out = tmp_path / "out"
        stats = remix_dataset(RemixRecipe(source=str(src)), out)
        assert stats["license_file"] is None
        assert not (out / "LICENSE").exists()

    def test_missing_template_fails_before_copying(self, esc, tmp_path) -> None:
        out = tmp_path / "out"
        with pytest.raises(NotFoundError, match="template"):
            remix_dataset(RemixRecipe(source=str(esc)), out, license_template=tmp_path / "no.txt")
        assert not out.exists()

    def test_license_template(self, esc, tmp_path) -> None:
        header = tmp_path / "header.txt"
        header.write_text("MY HEADER\n")
        out = tmp_path / "out"
        remix_dataset(RemixRecipe(source=str(esc), keep=["dog"]), out, license_template=header)
        assert (out / "LICENSE").read_text().startswith("MY HEADER")

    def test_name_defaults_to_dir(self, esc, tmp_path) -> None:
        stats = remix_dataset(RemixRecipe(source=str(esc)), tmp_path / "my-remix")
        assert stats["name"] == "my-remix"

    def test_metadata_is_valid_csv(self, esc, tmp_path) -> None:
        out = tmp_path / "out"
        remix_dataset(RemixRecipe(source=str(esc), keep=["cat"]), out)
        with (out / "metadata.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert all(r["file_name"].startswith("cat/") for r in rows)
