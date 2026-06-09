"""Tests for the canonical metadata-CSV schema, normalization, and I/O."""

from bioamla.datasets._metadata import (
    ATTRIBUTION_FIELDS,
    CORE_FIELDS,
    normalize_catalog_row,
    read_metadata_csv,
    write_metadata_csv,
)


class TestNormalizeCatalogRow:
    def test_xeno_canto_maps_to_canonical(self) -> None:
        row = {
            "file_name": "Turdus_migratorius/XC1_turdus.mp3",
            "xc_id": "1",
            "scientific_name": "Turdus migratorius",
            "common_name": "American Robin",
            "recordist": "Jane Doe",
            "url": "https://xeno-canto.org/1",
            "license": "cc-by",
            "quality": "A",
        }
        out = normalize_catalog_row(row, "xeno_canto")
        assert out["source"] == "xeno_canto"
        assert out["label"] == "Turdus migratorius"  # derived from scientific_name
        assert out["attribution"] == "Jane Doe"  # recordist -> attribution
        assert out["attr_url"] == "https://xeno-canto.org/1"  # url -> attr_url
        assert "recordist" not in out and "url" not in out  # remapped keys removed
        assert out["quality"] == "A"  # source-specific extra preserved
        assert out["license"] == "cc-by"  # core field untouched

    def test_macaulay_contributor_to_attribution(self) -> None:
        out = normalize_catalog_row(
            {"file_name": "f.wav", "scientific_name": "Corvus corax", "contributor": "Bob"},
            "macaulay",
        )
        assert out["source"] == "macaulay"
        assert out["attribution"] == "Bob"
        assert out["label"] == "Corvus corax"

    def test_existing_label_not_overwritten(self) -> None:
        out = normalize_catalog_row(
            {"file_name": "f.wav", "label": "explicit", "scientific_name": "X y"}, "inaturalist"
        )
        assert out["label"] == "explicit"


class TestWriteOrdering:
    def test_core_fields_lead_then_extras_sorted(self, tmp_path) -> None:
        path = tmp_path / "metadata.csv"
        rows = [
            normalize_catalog_row(
                {
                    "file_name": "a.wav",
                    "scientific_name": "Turdus migratorius",
                    "recordist": "Jane",
                    "xc_id": "1",
                    "quality": "A",
                },
                "xeno_canto",
            )
        ]
        write_metadata_csv(path, rows, merge_existing=False)
        header = path.read_text(encoding="utf-8").splitlines()[0].split(",")

        assert header[0] == "file_name"
        # Every present core/attribution column precedes every extra column.
        present_core = [c for c in CORE_FIELDS + ATTRIBUTION_FIELDS if c in header]
        extras = [c for c in header if c not in CORE_FIELDS + ATTRIBUTION_FIELDS]
        assert header[: len(present_core)] == present_core
        assert extras == sorted(extras)  # remainder alphabetical
        assert "xc_id" in extras and "quality" in extras

    def test_roundtrip_preserves_values(self, tmp_path) -> None:
        path = tmp_path / "metadata.csv"
        rows = [normalize_catalog_row({"file_name": "a.wav", "scientific_name": "X y"}, "macaulay")]
        write_metadata_csv(path, rows, merge_existing=False)
        read_rows, fieldnames = read_metadata_csv(path)
        assert read_rows[0]["file_name"] == "a.wav"
        assert read_rows[0]["source"] == "macaulay"
        assert read_rows[0]["label"] == "X y"
