"""Tests for catalogs.batch (thin wrappers over the generic batch engine)."""

from bioamla.catalogs import batch
from bioamla.catalogs._models import SpeciesInfo


class TestBatchDownloadXenoCanto:
    def test_invokes_download_per_species(self, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(batch.xeno_canto, "download", lambda **k: calls.append(k["species"]))
        result = batch.batch_download_xeno_canto(
            ["Turdus migratorius", "Strix varia"], output_dir="/tmp/xc"
        )
        assert result.total_files == 2
        assert result.successful == 2
        assert sorted(calls) == ["Strix varia", "Turdus migratorius"]

    def test_error_collected_when_continue_on_error(self, monkeypatch) -> None:
        def boom(**k):
            raise RuntimeError("api down")

        monkeypatch.setattr(batch.xeno_canto, "download", boom)
        result = batch.batch_download_xeno_canto(["A", "B"], continue_on_error=True)
        assert result.failed == 2
        assert len(result.errors) == 2


class TestBatchDownloadMacaulay:
    def test_invokes_download_per_code(self, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(batch.macaulay, "download", lambda **k: calls.append(k["species_code"]))
        result = batch.batch_download_macaulay(["amerob", "barswa"])
        assert result.successful == 2
        assert sorted(calls) == ["amerob", "barswa"]


class TestBatchLookupSpecies:
    def test_returns_scientific_names(self, monkeypatch) -> None:
        def fake_lookup(name, ebird_only=False):
            return SpeciesInfo(scientific_name=f"Sci {name}")

        monkeypatch.setattr(batch.species_mod, "lookup", fake_lookup)
        progress = []
        result = batch.batch_lookup_species(
            ["robin", "owl"], on_progress=lambda c, t: progress.append((c, t))
        )
        assert result.successful == 2
        assert "Sci robin" in result.output_files
        assert progress[-1] == (2, 2)

    def test_lookup_failure_collected(self, monkeypatch) -> None:
        from bioamla.exceptions import SpeciesError

        def boom(name, ebird_only=False):
            raise SpeciesError("not found")

        monkeypatch.setattr(batch.species_mod, "lookup", boom)
        result = batch.batch_lookup_species(["x"], continue_on_error=True)
        assert result.failed == 1
