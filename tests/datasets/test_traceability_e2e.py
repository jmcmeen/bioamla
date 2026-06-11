"""End-to-end: license traceability from catalog metadata -> clips -> ATTRIBUTIONS.md."""

import csv

import numpy as np

from bioamla.audio import save_audio
from bioamla.datasets import (
    Annotation,
    extract_labeled_dataset,
    generate_license_for_dataset,
    partition_dataset,
    save_bioamla_annotations,
)
from bioamla.datasets._metadata import write_metadata_csv


def test_license_survives_catalog_to_attributions(tmp_path):
    # 1. Fake catalog download: a recording + its annotations + a metadata.csv
    #    carrying license/attribution/source-url (xeno-canto shape).
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    sr = 16000
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False, dtype=np.float32)
    save_audio(str(catalog / "rec.wav"), 0.5 * np.sin(2 * np.pi * 440 * t), sr)
    save_bioamla_annotations(
        [
            Annotation(start_time=0.1, end_time=0.6, label="call"),
            Annotation(start_time=1.2, end_time=1.7, label="call"),
            Annotation(start_time=2.2, end_time=2.7, label="chorus"),
        ],
        str(catalog / "rec.json"),
    )
    write_metadata_csv(
        catalog / "metadata.csv",
        [
            {
                "file_name": "Hyla_cinerea/rec.wav",  # subdir form -> join by basename
                "source": "xeno_canto",
                "license": "CC-BY-NC",
                "attribution": "J. Doe",
                "attr_url": "https://xeno-canto.org/123",
            }
        ],
        merge_existing=False,
    )

    # 2. Extract clips (auto-detect the sibling metadata.csv).
    ds = tmp_path / "ds"
    result = extract_labeled_dataset(str(catalog), str(ds), layout="both")
    assert result["provenance"]["joined"] is True
    assert result["provenance"]["matched"] == 3

    # 3. Traceability assertion: every clip carries the source license + attribution.
    with (ds / "metadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    for row in rows:
        assert row["source_file"] == "rec.wav"
        assert row["source"] == "xeno_canto"
        assert row["license"] == "CC-BY-NC"
        assert row["attribution"] == "J. Doe"
        assert row["attr_url"] == "https://xeno-canto.org/123"

    # 4. Partition (subdirs) then generate the markdown attributions.
    partition_dataset(str(ds), splits=(0.6, 0.2, 0.2), seed=0, mode="subdirs", verbose=False)
    stats = generate_license_for_dataset(ds, format="md")

    # 5. The attributions file still traces every clip back to the source.
    md = (ds / "ATTRIBUTIONS.md").read_text(encoding="utf-8")
    assert "## xeno_canto — CC-BY-NC" in md
    assert "J. Doe" in md
    assert "https://xeno-canto.org/123" in md
    assert stats["attributions_count"] == 3
