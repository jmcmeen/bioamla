# models/xeno_canto.py
"""
Data models for Xeno-canto operations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import ToDictMixin


@dataclass
class XCRecording(ToDictMixin):
    """
    Information about a Xeno-canto recording.

    Attributes:
        id: Xeno-canto recording ID.
        genus: Genus name.
        species: Species epithet.
        subspecies: Subspecies name (if any).
        scientific_name: Full scientific name (Genus species).
        common_name: English common name.
        recordist: Name of the recordist.
        country: Country where recorded.
        location: Specific location.
        latitude: Latitude coordinate.
        longitude: Longitude coordinate.
        sound_type: Type of vocalization (song, call, etc.).
        quality: Recording quality (A, B, C, D, E).
        length: Recording length in format "m:ss".
        date: Recording date.
        time: Recording time.
        url: URL to the recording page.
        download_url: Direct URL to the audio file.
        license: License code.
        remarks: Recordist remarks.
    """

    id: str
    scientific_name: str
    common_name: str
    quality: str
    sound_type: str
    length: str
    location: str
    country: str
    recordist: str
    url: str
    download_url: str
    license: str
    genus: str = ""
    species: str = ""
    subspecies: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    date: str = ""
    time: str = ""
    remarks: str = ""

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "XCRecording":
        """Create a recording from Xeno-canto API response data."""
        lat = data.get("lat")
        lng = data.get("lng")

        genus = data.get("gen", "")
        species_epithet = data.get("sp", "")
        subspecies = data.get("ssp", "")

        # Build scientific name
        parts = [genus, species_epithet]
        if subspecies:
            parts.append(subspecies)
        scientific_name = " ".join(parts)

        return cls(
            id=data.get("id", ""),
            genus=genus,
            species=species_epithet,
            subspecies=subspecies,
            scientific_name=scientific_name,
            common_name=data.get("en", ""),
            recordist=data.get("rec", ""),
            country=data.get("cnt", ""),
            location=data.get("loc", ""),
            latitude=float(lat) if lat else None,
            longitude=float(lng) if lng else None,
            sound_type=data.get("type", ""),
            quality=data.get("q", ""),
            length=data.get("length", ""),
            date=data.get("date", ""),
            time=data.get("time", ""),
            url=data.get("url", ""),
            download_url=data.get("file", ""),
            license=data.get("lic", ""),
            remarks=data.get("rmk", ""),
        )


@dataclass
class SearchResult(ToDictMixin):
    """Result of a Xeno-canto search."""

    total_results: int
    recordings: List[XCRecording]
    query_params: Dict[str, Any]


@dataclass
class DownloadResult(ToDictMixin):
    """Result of a Xeno-canto download operation."""

    total: int
    downloaded: int
    failed: int
    skipped: int
    output_dir: str
    errors: List[str] = field(default_factory=list)
