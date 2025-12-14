"""
Dataset Exploration Utilities
=============================

This module provides data structures and utilities for exploring audio datasets
in the bioamla package. It supports loading audio file information, metadata,
and preparing data for display in the TUI dashboard.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torchaudio

from bioamla.core.globals import SUPPORTED_AUDIO_EXTENSIONS
from bioamla.core.metadata import read_metadata_csv


@dataclass
class AudioFileInfo:
    """Information about a single audio file."""

    path: Path
    filename: str
    size_bytes: int
    sample_rate: Optional[int] = None
    duration_seconds: Optional[float] = None
    num_channels: Optional[int] = None
    num_frames: Optional[int] = None
    format: Optional[str] = None
    # Metadata from CSV if available
    category: Optional[str] = None
    split: Optional[str] = None
    target: Optional[int] = None
    attribution: Optional[str] = None

    @property
    def size_human(self) -> str:
        """Return human-readable file size."""
        size = self.size_bytes
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    @property
    def duration_human(self) -> str:
        """Return human-readable duration."""
        if self.duration_seconds is None:
            return "Unknown"
        minutes = int(self.duration_seconds // 60)
        seconds = self.duration_seconds % 60
        if minutes > 0:
            return f"{minutes}m {seconds:.1f}s"
        return f"{seconds:.1f}s"


@dataclass
class DatasetInfo:
    """Information about an audio dataset directory."""

    path: Path
    name: str
    total_files: int = 0
    total_size_bytes: int = 0
    categories: Dict[str, int] = field(default_factory=dict)
    splits: Dict[str, int] = field(default_factory=dict)
    formats: Dict[str, int] = field(default_factory=dict)
    has_metadata: bool = False
    metadata_path: Optional[Path] = None

    @property
    def total_size_human(self) -> str:
        """Return human-readable total size."""
        size = self.total_size_bytes
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"


def get_audio_file_info(filepath: str, include_metadata: bool = True) -> AudioFileInfo:
    """
    Get detailed information about an audio file.

    Args:
        filepath: Path to the audio file
        include_metadata: Whether to load audio metadata (sample rate, duration, etc.)

    Returns:
        AudioFileInfo object with file details

    Raises:
        FileNotFoundError: If the file does not exist
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    info = AudioFileInfo(
        path=path,
        filename=path.name,
        size_bytes=path.stat().st_size,
        format=path.suffix.lower().lstrip("."),
    )

    if include_metadata:
        try:
            # Load audio to get metadata since torchaudio.info may not be available
            waveform, sample_rate = torchaudio.load(str(path))
            info.sample_rate = sample_rate
            info.num_channels = waveform.shape[0]
            info.num_frames = waveform.shape[1]
            if sample_rate and info.num_frames:
                info.duration_seconds = info.num_frames / sample_rate
        except Exception:
            # If we can't read metadata, just skip it
            pass

    return info


def scan_directory(
    directory: str,
    recursive: bool = True,
    load_audio_metadata: bool = False,
) -> Tuple[List[AudioFileInfo], DatasetInfo]:
    """
    Scan a directory for audio files and gather information.

    Args:
        directory: Path to the directory to scan
        recursive: Whether to scan subdirectories
        load_audio_metadata: Whether to load audio file metadata (slower but more detailed)

    Returns:
        Tuple of (list of AudioFileInfo, DatasetInfo)

    Raises:
        FileNotFoundError: If the directory does not exist
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    # Find audio files
    audio_files: List[AudioFileInfo] = []
    extensions = set(ext.lower() for ext in SUPPORTED_AUDIO_EXTENSIONS)

    if recursive:
        all_files = list(dir_path.rglob("*"))
    else:
        all_files = list(dir_path.glob("*"))

    for file_path in all_files:
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            try:
                file_info = get_audio_file_info(
                    str(file_path), include_metadata=load_audio_metadata
                )
                audio_files.append(file_info)
            except Exception:
                # Skip files we can't process
                continue

    # Load metadata if available
    metadata_path = dir_path / "metadata.csv"
    metadata_dict: Dict[str, dict] = {}
    has_metadata = metadata_path.exists()

    if has_metadata:
        rows, _ = read_metadata_csv(metadata_path)
        for row in rows:
            file_name = row.get("file_name", "")
            if file_name:
                # Handle both full paths and just filenames
                key = Path(file_name).name
                metadata_dict[key] = row

    # Enrich audio files with metadata
    for file_info in audio_files:
        if file_info.filename in metadata_dict:
            meta = metadata_dict[file_info.filename]
            file_info.category = meta.get("category")
            file_info.split = meta.get("split")
            try:
                file_info.target = int(meta.get("target", 0))
            except (ValueError, TypeError):
                pass
            file_info.attribution = meta.get("attr_id")

    # Build dataset info
    dataset_info = DatasetInfo(
        path=dir_path,
        name=dir_path.name,
        total_files=len(audio_files),
        has_metadata=has_metadata,
        metadata_path=metadata_path if has_metadata else None,
    )

    # Aggregate statistics
    for file_info in audio_files:
        dataset_info.total_size_bytes += file_info.size_bytes

        # Count by format
        if file_info.format:
            fmt = file_info.format.upper()
            dataset_info.formats[fmt] = dataset_info.formats.get(fmt, 0) + 1

        # Count by category
        if file_info.category:
            dataset_info.categories[file_info.category] = (
                dataset_info.categories.get(file_info.category, 0) + 1
            )

        # Count by split
        if file_info.split:
            dataset_info.splits[file_info.split] = (
                dataset_info.splits.get(file_info.split, 0) + 1
            )

    return audio_files, dataset_info


def get_category_summary(audio_files: List[AudioFileInfo]) -> Dict[str, Dict]:
    """
    Get a summary of audio files grouped by category.

    Args:
        audio_files: List of AudioFileInfo objects

    Returns:
        Dictionary mapping category names to summary statistics
    """
    categories: Dict[str, Dict] = {}

    for file_info in audio_files:
        cat = file_info.category or "Uncategorized"
        if cat not in categories:
            categories[cat] = {
                "count": 0,
                "total_size": 0,
                "total_duration": 0.0,
                "files": [],
            }

        categories[cat]["count"] += 1
        categories[cat]["total_size"] += file_info.size_bytes
        if file_info.duration_seconds:
            categories[cat]["total_duration"] += file_info.duration_seconds
        categories[cat]["files"].append(file_info)

    return categories


def get_split_summary(audio_files: List[AudioFileInfo]) -> Dict[str, Dict]:
    """
    Get a summary of audio files grouped by split.

    Args:
        audio_files: List of AudioFileInfo objects

    Returns:
        Dictionary mapping split names to summary statistics
    """
    splits: Dict[str, Dict] = {}

    for file_info in audio_files:
        split = file_info.split or "Unknown"
        if split not in splits:
            splits[split] = {
                "count": 0,
                "total_size": 0,
                "files": [],
            }

        splits[split]["count"] += 1
        splits[split]["total_size"] += file_info.size_bytes
        splits[split]["files"].append(file_info)

    return splits


def filter_audio_files(
    audio_files: List[AudioFileInfo],
    category: Optional[str] = None,
    split: Optional[str] = None,
    format: Optional[str] = None,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    search_term: Optional[str] = None,
) -> List[AudioFileInfo]:
    """
    Filter audio files based on criteria.

    Args:
        audio_files: List of AudioFileInfo objects to filter
        category: Filter by category name
        split: Filter by split name
        format: Filter by audio format (e.g., 'wav', 'mp3')
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        search_term: Search term for filename matching

    Returns:
        Filtered list of AudioFileInfo objects
    """
    result = audio_files

    if category:
        result = [f for f in result if f.category == category]

    if split:
        result = [f for f in result if f.split == split]

    if format:
        fmt = format.lower().lstrip(".")
        result = [f for f in result if f.format == fmt]

    if min_duration is not None:
        result = [
            f for f in result
            if f.duration_seconds is not None and f.duration_seconds >= min_duration
        ]

    if max_duration is not None:
        result = [
            f for f in result
            if f.duration_seconds is not None and f.duration_seconds <= max_duration
        ]

    if search_term:
        term = search_term.lower()
        result = [f for f in result if term in f.filename.lower()]

    return result


def sort_audio_files(
    audio_files: List[AudioFileInfo],
    sort_by: str = "name",
    reverse: bool = False,
) -> List[AudioFileInfo]:
    """
    Sort audio files by a given field.

    Args:
        audio_files: List of AudioFileInfo objects to sort
        sort_by: Field to sort by ('name', 'size', 'duration', 'category', 'format')
        reverse: Whether to reverse the sort order

    Returns:
        Sorted list of AudioFileInfo objects
    """
    key_funcs = {
        "name": lambda f: f.filename.lower(),
        "size": lambda f: f.size_bytes,
        "duration": lambda f: f.duration_seconds or 0,
        "category": lambda f: (f.category or "").lower(),
        "format": lambda f: (f.format or "").lower(),
    }

    key_func = key_funcs.get(sort_by, key_funcs["name"])
    return sorted(audio_files, key=key_func, reverse=reverse)
