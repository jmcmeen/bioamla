"""Tests for iNaturalist catalog (helpers, validation, errors; no network)."""

import csv

import pytest

from bioamla.catalogs import inat
from bioamla.catalogs._metadata import (
    get_existing_observation_ids,
    read_metadata_csv,
    write_metadata_csv,
)
from bioamla.catalogs._models import (
    INaturalistDownloadResult,
    INaturalistSearchResult,
    ObservationInfo,
    ProjectStats,
    TaxonInfo,
)
from bioamla.exceptions import CatalogError, InvalidInputError


class TestLoadTaxonIdsFromCsv:
    def test_valid_csv(self, tmp_path) -> None:
        csv_path = tmp_path / "taxa.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["taxon_id"])
            writer.writeheader()
            writer.writerow({"taxon_id": "3"})
            writer.writerow({"taxon_id": "19893"})
        assert inat._load_taxon_ids_from_csv(csv_path) == [3, 19893]

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(InvalidInputError):
            inat._load_taxon_ids_from_csv(tmp_path / "nope.csv")

    def test_missing_column_raises(self, tmp_path) -> None:
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("name\nfoo\n")
        with pytest.raises(InvalidInputError):
            inat._load_taxon_ids_from_csv(csv_path)

    def test_no_valid_ids_raises(self, tmp_path) -> None:
        csv_path = tmp_path / "empty.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["taxon_id"])
            writer.writeheader()
            writer.writerow({"taxon_id": ""})
        with pytest.raises(InvalidInputError):
            inat._load_taxon_ids_from_csv(csv_path)


class TestSearch:
    def test_search_parses_results(self, monkeypatch) -> None:
        monkeypatch.setattr(
            inat, "get_observations", lambda **k: {"results": [{"id": 1}, {"id": 2}]}
        )
        result = inat.search(taxon_name="Strix varia")
        assert isinstance(result, INaturalistSearchResult)
        assert result.total_results == 2

    def test_search_failure_raises_catalog_error(self, monkeypatch) -> None:
        def boom(**k):
            raise RuntimeError("api down")

        monkeypatch.setattr(inat, "get_observations", boom)
        with pytest.raises(CatalogError):
            inat.search(taxon_name="Strix varia")


class TestGetTaxa:
    def test_requires_place_or_project(self) -> None:
        with pytest.raises(InvalidInputError):
            inat.get_taxa()

    def test_returns_sorted_taxa(self, monkeypatch) -> None:
        pages = [
            {
                "results": [
                    {"taxon": {"id": 1, "name": "A", "preferred_common_name": "a"}, "count": 5},
                    {"taxon": {"id": 2, "name": "B", "preferred_common_name": "b"}, "count": 50},
                ]
            },
            {"results": []},
        ]
        calls = {"n": 0}

        def fake(**k):
            i = calls["n"]
            calls["n"] += 1
            return pages[i]

        monkeypatch.setattr(inat, "get_observation_species_counts", fake)
        taxa = inat.get_taxa(place_id=1)
        assert isinstance(taxa[0], TaxonInfo)
        # Sorted by observation_count desc.
        assert taxa[0].observation_count == 50

    def test_failure_raises_catalog_error(self, monkeypatch) -> None:
        def boom(**k):
            raise RuntimeError("x")

        monkeypatch.setattr(inat, "get_observation_species_counts", boom)
        with pytest.raises(CatalogError):
            inat.get_taxa(place_id=1)


class TestProjectStats:
    def test_project_not_found_raises(self, monkeypatch) -> None:
        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"results": []}

        monkeypatch.setattr(inat.requests, "get", lambda *a, **k: FakeResp())
        with pytest.raises(CatalogError):
            inat.get_project_stats("missing-project")

    def test_success(self, monkeypatch) -> None:
        responses = iter(
            [
                _FakeResponse(
                    json_data={
                        "results": [
                            {
                                "id": 5,
                                "title": "Owls",
                                "slug": "owls",
                                "created_at": "2020",
                                "project_type": "collection",
                                "place": {"display_name": "USA"},
                            }
                        ]
                    }
                ),
                _FakeResponse(json_data={"total_results": 100}),
                _FakeResponse(json_data={"total_results": 12}),
                _FakeResponse(json_data={"total_results": 7}),
            ]
        )
        monkeypatch.setattr(inat.requests, "get", lambda *a, **k: next(responses))
        stats = inat.get_project_stats("owls")
        assert isinstance(stats, ProjectStats)
        assert stats.observation_count == 100
        assert stats.species_count == 12
        assert stats.observers_count == 7
        assert stats.place == "USA"

    def test_request_failure_raises_catalog_error(self, monkeypatch) -> None:
        def boom(*a, **k):
            raise RuntimeError("down")

        monkeypatch.setattr(inat.requests, "get", boom)
        with pytest.raises(CatalogError):
            inat.get_project_stats("owls")


class TestMetadataHelpers:
    def test_write_then_read_roundtrip(self, tmp_path) -> None:
        path = tmp_path / "metadata.csv"
        rows = [
            {
                "file_name": "inat_1_sound_2.mp3",
                "split": "train",
                "target": "",
                "label": "strix_varia",
                "attr_id": "user",
                "attr_lic": "cc-by",
                "attr_url": "http://x",
                "attr_note": "",
            }
        ]
        written = write_metadata_csv(path, rows, merge_existing=False)
        assert written == 1
        read_rows, fieldnames = read_metadata_csv(path)
        assert len(read_rows) == 1
        assert "file_name" in fieldnames
        # Required fields should come first.
        assert read_rows[0]["file_name"] == "inat_1_sound_2.mp3"

    def test_get_existing_observation_ids(self, tmp_path) -> None:
        path = tmp_path / "metadata.csv"
        write_metadata_csv(
            path,
            [
                {
                    "file_name": "strix_varia/inat_111_sound_222.mp3",
                    "split": "train",
                    "target": "",
                    "label": "strix_varia",
                    "attr_id": "u",
                    "attr_lic": "cc0",
                    "attr_url": "x",
                    "attr_note": "",
                }
            ],
            merge_existing=False,
        )
        existing = get_existing_observation_ids(path)
        assert (111, 222) in existing

    def test_read_missing_file_returns_empty(self, tmp_path) -> None:
        rows, fieldnames = read_metadata_csv(tmp_path / "nope.csv")
        assert rows == []
        assert fieldnames == set()


class _FakeResponse:
    """Minimal stand-in for requests.Response used by _download_file/project stats."""

    def __init__(self, *, headers=None, chunks=(b"abc",), json_data=None, raise_exc=None):
        self.headers = headers or {}
        self._chunks = chunks
        self._json = json_data or {}
        self._raise_exc = raise_exc

    def raise_for_status(self):
        if self._raise_exc is not None:
            raise self._raise_exc

    def iter_content(self, chunk_size=8192):
        yield from self._chunks

    def json(self):
        return self._json


# =============================================================================
# _download_file
# =============================================================================


class TestDownloadFile:
    def test_success_writes_file(self, tmp_path, monkeypatch) -> None:
        resp = _FakeResponse(chunks=(b"hello", b"world"))
        monkeypatch.setattr(inat.requests, "get", lambda *a, **k: resp)
        target = tmp_path / "out.mp3"
        assert inat._download_file("http://x/file.mp3", target) is True
        assert target.read_bytes() == b"helloworld"

    def test_content_type_adjusts_extension(self, tmp_path, monkeypatch) -> None:
        resp = _FakeResponse(headers={"Content-Type": "audio/mpeg"}, chunks=(b"x",))
        monkeypatch.setattr(inat.requests, "get", lambda *a, **k: resp)
        # No extension -> helper derives one from content type.
        target = tmp_path / "noext"
        assert inat._download_file("http://x/noext", target) is True
        # Some file with an mp3 suffix should now exist in the dir.
        assert any(p.suffix == ".mp3" for p in tmp_path.iterdir())

    def test_request_exception_returns_false(self, tmp_path, monkeypatch) -> None:
        def boom(*a, **k):
            raise inat.requests.RequestException("network down")

        monkeypatch.setattr(inat.requests, "get", boom)
        assert inat._download_file("http://x", tmp_path / "f.mp3", verbose=True) is False


# =============================================================================
# _discover_taxa_from_query
# =============================================================================


class TestDiscoverTaxa:
    def test_collects_ids_across_pages(self, monkeypatch) -> None:
        pages = [
            {"results": [{"taxon": {"id": 1}}, {"taxon": {"id": 2}}] + [{"taxon": {}}] * 498},
            {"results": [{"taxon": {"id": 3}}]},
        ]
        calls = {"n": 0}

        def fake(**k):
            i = calls["n"]
            calls["n"] += 1
            return pages[i]

        monkeypatch.setattr(inat, "get_observation_species_counts", fake)
        ids = inat._discover_taxa_from_query(taxon_name="Aves")
        assert ids[:2] == [1, 2]
        assert 3 in ids

    def test_failure_returns_empty(self, monkeypatch) -> None:
        def boom(**k):
            raise RuntimeError("x")

        monkeypatch.setattr(inat, "get_observation_species_counts", boom)
        assert inat._discover_taxa_from_query(taxon_name="Aves") == []


# =============================================================================
# download
# =============================================================================


def _obs(obs_id=10, sound_id=99, name="Strix varia", rank="species", sounds=None):
    if sounds is None:
        sounds = [
            {
                "id": sound_id,
                "file_url": f"http://x/{sound_id}.mp3",
                "license_code": "cc-by",
            }
        ]
    return {
        "id": obs_id,
        "sounds": sounds,
        "taxon": {"id": 7, "name": name, "preferred_common_name": "Barred Owl", "rank": rank},
        "observed_on": "2024-01-01",
        "user": {"login": "alice"},
        "quality_grade": "research",
    }


def _once(results):
    """get_observations stub: return ``results`` on the first call, empty after.

    Mirrors real pagination so download()'s page loop terminates even when no
    observation is successfully processed (failed/skipped downloads).
    """
    state = {"called": False}

    def fake(**k):
        if state["called"]:
            return {"results": []}
        state["called"] = True
        return {"results": results}

    return fake


class TestDownload:
    def test_downloads_and_writes_metadata(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(inat, "get_observations", lambda **k: {"results": [_obs()]})
        monkeypatch.setattr(inat, "_download_file", lambda url, fp, verbose=True: True)
        monkeypatch.setattr(inat.time, "sleep", lambda s: None)

        seen: list[tuple[int, int, str]] = []
        result = inat.download(
            output_dir=str(tmp_path),
            taxon_ids=[7],
            obs_per_taxon=1,
            include_metadata=True,
            progress_callback=lambda c, t, f: seen.append((c, t, f)),
        )
        assert isinstance(result, INaturalistDownloadResult)
        assert result.total_sounds == 1
        assert result.total_observations == 1
        assert seen  # progress callback invoked
        meta = tmp_path / "metadata.csv"
        assert meta.exists()
        rows = list(csv.DictReader(meta.open()))
        assert rows[0]["label"] == "strix_varia"

    def test_failed_download_recorded(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(inat, "get_observations", _once([_obs()]))
        monkeypatch.setattr(inat, "_download_file", lambda url, fp, verbose=True: False)
        monkeypatch.setattr(inat.time, "sleep", lambda s: None)
        result = inat.download(output_dir=str(tmp_path), taxon_ids=[7], obs_per_taxon=1)
        assert result.failed_downloads == 1
        assert result.errors

    def test_subspecies_resolves_to_species_via_ancestors(self, tmp_path, monkeypatch) -> None:
        obs = _obs(name="Strix varia georgica", rank="subspecies")
        obs["taxon"]["ancestors"] = [{"rank": "species", "name": "Strix varia"}]
        monkeypatch.setattr(inat, "get_observations", lambda **k: {"results": [obs]})
        written: list[str] = []
        monkeypatch.setattr(
            inat, "_download_file", lambda url, fp, verbose=True: written.append(str(fp)) or True
        )
        monkeypatch.setattr(inat.time, "sleep", lambda s: None)
        inat.download(output_dir=str(tmp_path), taxon_ids=[7], obs_per_taxon=1)
        assert any("strix_varia" in w for w in written)

    def test_no_taxa_discovers_then_falls_back_to_none(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(inat, "_discover_taxa_from_query", lambda **k: [])
        monkeypatch.setattr(inat, "get_observations", lambda **k: {"results": []})
        result = inat.download(output_dir=str(tmp_path), taxon_name="Aves", obs_per_taxon=1)
        assert result.total_sounds == 0

    def test_skips_existing(self, tmp_path, monkeypatch) -> None:
        # Pre-seed an existing metadata row so (obs_id, sound_id) is skipped.
        from bioamla.catalogs._metadata import write_metadata_csv

        write_metadata_csv(
            tmp_path / "metadata.csv",
            [
                {
                    "file_name": "strix_varia/inat_10_sound_99.mp3",
                    "split": "train",
                    "target": "",
                    "label": "strix_varia",
                    "attr_id": "alice",
                    "attr_lic": "cc-by",
                    "attr_url": "http://x/99.mp3",
                    "attr_note": "",
                }
            ],
            merge_existing=False,
        )
        monkeypatch.setattr(inat, "get_observations", _once([_obs()]))
        monkeypatch.setattr(inat.time, "sleep", lambda s: None)
        result = inat.download(output_dir=str(tmp_path), taxon_ids=[7], obs_per_taxon=1)
        assert result.skipped_existing == 1
        assert result.total_sounds == 0

    def test_taxon_csv_loaded(self, tmp_path, monkeypatch) -> None:
        csv_path = tmp_path / "taxa.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["taxon_id"])
            w.writeheader()
            w.writerow({"taxon_id": "7"})
        monkeypatch.setattr(inat, "get_observations", lambda **k: {"results": []})
        result = inat.download(
            output_dir=str(tmp_path / "out"), taxon_csv=str(csv_path), obs_per_taxon=1
        )
        assert result.total_sounds == 0

    def test_invalid_taxon_csv_raises(self, tmp_path) -> None:
        with pytest.raises(InvalidInputError):
            inat.download(output_dir=str(tmp_path / "out"), taxon_csv=str(tmp_path / "nope.csv"))

    def test_api_failure_raises_catalog_error(self, tmp_path, monkeypatch) -> None:
        def boom(**k):
            raise RuntimeError("api down")

        monkeypatch.setattr(inat, "get_observations", boom)
        with pytest.raises(CatalogError):
            inat.download(output_dir=str(tmp_path), taxon_ids=[7], obs_per_taxon=1)

    def test_sound_license_normalized(self, tmp_path, monkeypatch) -> None:
        captured = {}

        def fake_get(**k):
            captured["sound_license"] = k.get("sound_license")
            return {"results": []}

        monkeypatch.setattr(inat, "get_observations", fake_get)
        inat.download(
            output_dir=str(tmp_path),
            taxon_ids=[7],
            sound_license=["cc-by", "cc0"],
            obs_per_taxon=1,
        )
        assert captured["sound_license"] == ["CC-BY", "CC0"]


# =============================================================================
# download_from_observations
# =============================================================================


class TestDownloadFromObservations:
    def test_downloads_specific_observations(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(inat, "get_observations", lambda **k: {"results": [_obs()]})
        monkeypatch.setattr(inat, "_download_file", lambda url, fp, verbose=True: True)
        result = inat.download_from_observations([10], str(tmp_path))
        assert result.total_sounds == 1
        assert result.total_observations == 1

    def test_observation_not_found_recorded(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(inat, "get_observations", lambda **k: {"results": []})
        result = inat.download_from_observations([404], str(tmp_path))
        assert result.failed_downloads == 0
        assert any("not found" in e for e in result.errors)

    def test_per_observation_exception_recorded(self, tmp_path, monkeypatch) -> None:
        def boom(**k):
            raise RuntimeError("boom")

        monkeypatch.setattr(inat, "get_observations", boom)
        result = inat.download_from_observations([1], str(tmp_path))
        assert result.failed_downloads == 1
        assert result.errors

    def test_download_failure_increments_failed(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(inat, "get_observations", lambda **k: {"results": [_obs()]})
        monkeypatch.setattr(inat, "_download_file", lambda url, fp, verbose=True: False)
        result = inat.download_from_observations([10], str(tmp_path), organize_by_taxon=False)
        assert result.failed_downloads == 1


# =============================================================================
# get_observation_info
# =============================================================================


class TestObservationInfo:
    def test_success(self, monkeypatch) -> None:
        monkeypatch.setattr(inat, "get_observations", lambda **k: {"results": [_obs()]})
        info = inat.get_observation_info(10)
        assert isinstance(info, ObservationInfo)
        assert info.id == 10
        assert info.taxon_name == "Strix varia"

    def test_not_found_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(inat, "get_observations", lambda **k: {"results": []})
        with pytest.raises(CatalogError):
            inat.get_observation_info(404)

    def test_api_failure_raises(self, monkeypatch) -> None:
        def boom(**k):
            raise RuntimeError("x")

        monkeypatch.setattr(inat, "get_observations", boom)
        with pytest.raises(CatalogError):
            inat.get_observation_info(10)
