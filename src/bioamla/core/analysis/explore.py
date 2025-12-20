"""
Dataset Exploration Utilities
=============================

This module provides data structures and utilities for exploring audio datasets
in the bioamla package. It supports loading audio file information, metadata,
and preparing data for display in the TUI dashboard.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torchaudio

from bioamla.core.globals import SUPPORTED_AUDIO_EXTENSIONS
from bioamla.core.metadata import read_metadata_csv


@dataclass
class AudioFileInfo:
    """
    Information about a single audio file.

    Attributes:
        path: Full path to the audio file.
        filename: Name of the file (without directory path).
        size_bytes: File size in bytes.
        sample_rate: Audio sample rate in Hz.
        duration_seconds: Duration of the audio in seconds.
        num_channels: Number of audio channels.
        num_frames: Total number of audio frames/samples.
        format: Audio file format (e.g., 'wav', 'mp3').
        label: Label/class name from metadata CSV.
        split: Dataset split (e.g., 'train', 'test') from metadata CSV.
        target: Numeric target/label ID from metadata CSV.
        attribution: Attribution identifier from metadata CSV.
    """

    path: Path
    filename: str
    size_bytes: int
    sample_rate: Optional[int] = None
    duration_seconds: Optional[float] = None
    num_channels: Optional[int] = None
    num_frames: Optional[int] = None
    format: Optional[str] = None
    label: Optional[str] = None
    split: Optional[str] = None
    target: Optional[int] = None
    attribution: Optional[str] = None

    @property
    def size_human(self) -> str:
        """Return human-readable file size (e.g., '1.5 MB')."""
        size = self.size_bytes
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    @property
    def duration_human(self) -> str:
        """Return human-readable duration (e.g., '2m 30.5s' or '15.3s')."""
        if self.duration_seconds is None:
            return "Unknown"
        minutes = int(self.duration_seconds // 60)
        seconds = self.duration_seconds % 60
        if minutes > 0:
            return f"{minutes}m {seconds:.1f}s"
        return f"{seconds:.1f}s"


@dataclass
class DatasetInfo:
    """
    Information about an audio dataset directory.

    Attributes:
        path: Full path to the dataset directory.
        name: Name of the dataset directory.
        total_files: Total number of audio files in the dataset.
        total_size_bytes: Total size of all audio files in bytes.
        labels: Mapping of label names to file counts.
        splits: Mapping of split names to file counts.
        formats: Mapping of audio formats to file counts.
        has_metadata: Whether a metadata.csv file was found.
        metadata_path: Path to the metadata.csv file if it exists.
    """

    path: Path
    name: str
    total_files: int = 0
    total_size_bytes: int = 0
    labels: Dict[str, int] = field(default_factory=dict)
    splits: Dict[str, int] = field(default_factory=dict)
    formats: Dict[str, int] = field(default_factory=dict)
    has_metadata: bool = False
    metadata_path: Optional[Path] = None

    @property
    def total_size_human(self) -> str:
        """Return human-readable total size (e.g., '1.5 GB')."""
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

    Searches the specified directory for audio files with supported extensions,
    optionally loading metadata from a metadata.csv file if present.

    Args:
        directory: Path to the directory to scan.
        recursive: Whether to scan subdirectories recursively.
        load_audio_metadata: Whether to load audio file metadata (sample rate,
            duration, etc.). This is slower but provides more detailed info.

    Returns:
        Tuple containing:
            - List of AudioFileInfo objects for each audio file found.
            - DatasetInfo object with aggregated statistics.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If the path is not a directory.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    # Find audio files
    audio_files: List[AudioFileInfo] = []
    extensions = {ext.lower() for ext in SUPPORTED_AUDIO_EXTENSIONS}

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
            file_info.label = meta.get("label")
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

        # Count by label
        if file_info.label:
            dataset_info.labels[file_info.label] = dataset_info.labels.get(file_info.label, 0) + 1

        # Count by split
        if file_info.split:
            dataset_info.splits[file_info.split] = dataset_info.splits.get(file_info.split, 0) + 1

    return audio_files, dataset_info


def get_label_summary(audio_files: List[AudioFileInfo]) -> Dict[str, Dict[str, any]]:
    """
    Get a summary of audio files grouped by label.

    Groups audio files by their label and computes aggregate
    statistics for each label.

    Args:
        audio_files: List of AudioFileInfo objects to summarize.

    Returns:
        Dictionary mapping label names to summary dictionaries containing:
            - count: Number of files with the label.
            - total_size: Total size in bytes.
            - total_duration: Total duration in seconds.
            - files: List of AudioFileInfo objects with the label.
    """
    labels: Dict[str, Dict] = {}

    for file_info in audio_files:
        lbl = file_info.label or "Unlabeled"
        if lbl not in labels:
            labels[lbl] = {
                "count": 0,
                "total_size": 0,
                "total_duration": 0.0,
                "files": [],
            }

        labels[lbl]["count"] += 1
        labels[lbl]["total_size"] += file_info.size_bytes
        if file_info.duration_seconds:
            labels[lbl]["total_duration"] += file_info.duration_seconds
        labels[lbl]["files"].append(file_info)

    return labels


def get_split_summary(audio_files: List[AudioFileInfo]) -> Dict[str, Dict[str, any]]:
    """
    Get a summary of audio files grouped by dataset split.

    Groups audio files by their split label (e.g., 'train', 'test', 'val')
    and computes aggregate statistics for each split.

    Args:
        audio_files: List of AudioFileInfo objects to summarize.

    Returns:
        Dictionary mapping split names to summary dictionaries containing:
            - count: Number of files in the split.
            - total_size: Total size in bytes.
            - files: List of AudioFileInfo objects in the split.
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
    label: Optional[str] = None,
    split: Optional[str] = None,
    format: Optional[str] = None,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    search_term: Optional[str] = None,
) -> List[AudioFileInfo]:
    """
    Filter audio files based on specified criteria.

    Applies one or more filters to a list of audio files. All specified
    criteria must match (AND logic) for a file to be included in the result.

    Args:
        audio_files: List of AudioFileInfo objects to filter.
        label: Filter by exact label name match.
        split: Filter by exact split name match.
        format: Filter by audio format (e.g., 'wav', 'mp3'). Case-insensitive.
        min_duration: Minimum duration in seconds (inclusive).
        max_duration: Maximum duration in seconds (inclusive).
        search_term: Case-insensitive substring search in filename.

    Returns:
        List of AudioFileInfo objects matching all specified criteria.
    """
    result = audio_files

    if label:
        result = [f for f in result if f.label == label]

    if split:
        result = [f for f in result if f.split == split]

    if format:
        fmt = format.lower().lstrip(".")
        result = [f for f in result if f.format == fmt]

    if min_duration is not None:
        result = [
            f
            for f in result
            if f.duration_seconds is not None and f.duration_seconds >= min_duration
        ]

    if max_duration is not None:
        result = [
            f
            for f in result
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
    Sort audio files by a specified field.

    Returns a new sorted list without modifying the original.

    Args:
        audio_files: List of AudioFileInfo objects to sort.
        sort_by: Field to sort by. Valid options:
            - 'name': Sort by filename (case-insensitive).
            - 'size': Sort by file size in bytes.
            - 'duration': Sort by audio duration.
            - 'label': Sort by label (case-insensitive).
            - 'format': Sort by audio format (case-insensitive).
        reverse: If True, sort in descending order.

    Returns:
        New list of AudioFileInfo objects in sorted order.
    """
    key_funcs = {
        "name": lambda f: f.filename.lower(),
        "size": lambda f: f.size_bytes,
        "duration": lambda f: f.duration_seconds or 0,
        "label": lambda f: (f.label or "").lower(),
        "format": lambda f: (f.format or "").lower(),
    }

    key_func = key_funcs.get(sort_by, key_funcs["name"])
    return sorted(audio_files, key=key_func, reverse=reverse)
