"""Tests for the provenance-tolerant license/attribution generator."""

import pytest

from bioamla.datasets import generate_license_for_dataset
from bioamla.datasets._metadata import write_metadata_csv
from bioamla.exceptions import InvalidInputError


def _dataset(tmp_path, rows):
    ds = tmp_path / "ds"
    ds.mkdir()
    write_metadata_csv(ds / "metadata.csv", rows, merge_existing=False)
    return ds


def _license_text(ds):
    return (ds / "LICENSE").read_text(encoding="utf-8")


class TestProvenanceTolerantParser:
    def test_xeno_canto_canonical_fields(self, tmp_path):
        # xeno-canto: license + attribution + attr_url, NO attr_lic/attr_id.
        ds = _dataset(
            tmp_path,
            [
                {
                    "file_name": "rec.wav",
                    "source": "xeno_canto",
                    "license": "CC-BY-NC-SA",
                    "attribution": "Recordist Name",
                    "attr_url": "https://xeno-canto.org/123",
                }
            ],
        )
        stats = generate_license_for_dataset(ds)
        assert stats["attributions_count"] == 1
        text = _license_text(ds)
        assert "CC-BY-NC-SA" in text
        assert "Recordist Name" in text
        assert "https://xeno-canto.org/123" in text
        assert "xeno_canto" in text

    def test_macaulay_attribution_only(self, tmp_path):
        # macaulay: only attribution (no license) — still a valid signal.
        ds = _dataset(
            tmp_path,
            [{"file_name": "r.wav", "source": "macaulay", "attribution": "Contributor"}],
        )
        stats = generate_license_for_dataset(ds)
        assert stats["attributions_count"] == 1
        assert "Contributor" in _license_text(ds)

    def test_inaturalist_attr_block_still_works(self, tmp_path):
        ds = _dataset(
            tmp_path,
            [
                {
                    "file_name": "a.wav",
                    "attr_id": "user1",
                    "attr_lic": "CC0",
                    "attr_url": "http://inat/1",
                    "attr_note": "obs",
                }
            ],
        )
        stats = generate_license_for_dataset(ds)
        assert stats["attributions_count"] == 1
        text = _license_text(ds)
        assert "CC0" in text and "user1" in text

    def test_bare_metadata_raises(self, tmp_path):
        ds = _dataset(tmp_path, [{"file_name": "a.wav", "label": "call"}])
        with pytest.raises(InvalidInputError):
            generate_license_for_dataset(ds)

    def test_mixed_sources_all_listed(self, tmp_path):
        ds = _dataset(
            tmp_path,
            [
                {"file_name": "xc.wav", "source": "xeno_canto", "license": "CC-BY"},
                {"file_name": "ml.wav", "source": "macaulay", "attribution": "Bob"},
                {"file_name": "inat.wav", "attr_id": "u", "attr_lic": "CC0"},
            ],
        )
        stats = generate_license_for_dataset(ds)
        assert stats["attributions_count"] == 3
        text = _license_text(ds)
        assert "xc.wav" in text and "ml.wav" in text and "inat.wav" in text

    def test_rows_without_signal_are_skipped(self, tmp_path):
        # One row has a license, the other has nothing -> only the first counts.
        ds = _dataset(
            tmp_path,
            [
                {"file_name": "a.wav", "license": "CC-BY"},
                {"file_name": "b.wav", "label": "x"},
            ],
        )
        stats = generate_license_for_dataset(ds)
        assert stats["attributions_count"] == 1
        text = _license_text(ds)
        assert "a.wav" in text and "b.wav" not in text
