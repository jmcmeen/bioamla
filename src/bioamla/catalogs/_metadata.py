"""Back-compat shim: catalog metadata helpers now live in datasets._metadata.

The canonical metadata-CSV schema and helpers were consolidated into
:mod:`bioamla.datasets._metadata` (one source of truth for the group-level
``metadata.csv``). This module re-exports them so existing catalog imports keep
working.
"""

from bioamla.datasets._metadata import (
    ATTRIBUTION_FIELDS,
    CORE_FIELDS,
    OPTIONAL_INAT_FIELDS,
    REQUIRED_FIELDS,
    get_existing_observation_ids,
    normalize_catalog_row,
    read_metadata_csv,
    write_metadata_csv,
)

__all__ = [
    "read_metadata_csv",
    "write_metadata_csv",
    "normalize_catalog_row",
    "get_existing_observation_ids",
    "CORE_FIELDS",
    "ATTRIBUTION_FIELDS",
    "OPTIONAL_INAT_FIELDS",
    "REQUIRED_FIELDS",
]
