"""
Unit tests for bioamla.core.fileutils module.
"""


from bioamla.core.fileutils import (
    ensure_directory,
    find_species_name,
    get_extension_from_content_type,
    get_extension_from_url,
    get_relative_path,
    sanitize_filename,
)


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_basic_sanitization(self):
        """Test basic filename sanitization."""
        assert sanitize_filename("My Species Name") == "my_species_name"

    def test_removes_invalid_characters(self):
        """Test removal of invalid characters."""
        assert sanitize_filename('Test: File?') == "test__file_"
        assert sanitize_filename('Path/To\\File') == "path_to_file"
        assert sanitize_filename('Name<With>Brackets') == "name_with_brackets"

    def test_empty_string_returns_unknown(self):
        """Test that empty string returns 'unknown'."""
        assert sanitize_filename("") == "unknown"

    def test_strips_periods_and_spaces(self):
        """Test stripping of leading/trailing periods and spaces."""
        # Note: spaces are replaced with underscores first, then periods stripped
        assert sanitize_filename("..test..") == "test"
        assert sanitize_filename(". .") == "_"  # space -> underscore, periods stripped
        assert sanitize_filename("...") == "unknown"  # all periods -> empty -> unknown

    def test_lowercases_result(self):
        """Test that result is lowercase."""
        assert sanitize_filename("UPPERCASE") == "uppercase"
        assert sanitize_filename("MixedCase") == "mixedcase"

    def test_replaces_spaces_with_underscores(self):
        """Test that spaces are replaced with underscores."""
        assert sanitize_filename("word one two") == "word_one_two"

    def test_scientific_names(self):
        """Test sanitization of scientific names."""
        assert sanitize_filename("Lithobates catesbeianus") == "lithobates_catesbeianus"
        assert sanitize_filename("Strix varia") == "strix_varia"


class TestGetExtensionFromUrl:
    """Tests for get_extension_from_url function."""

    def test_wav_extension(self):
        """Test detection of .wav extension."""
        assert get_extension_from_url("https://example.com/audio.wav") == ".wav"
        assert get_extension_from_url("https://example.com/audio.WAV") == ".wav"

    def test_mp3_extension(self):
        """Test detection of .mp3 extension."""
        assert get_extension_from_url("https://example.com/audio.mp3") == ".mp3"

    def test_m4a_extension(self):
        """Test detection of .m4a extension."""
        assert get_extension_from_url("https://example.com/audio.m4a") == ".m4a"

    def test_ogg_extension(self):
        """Test detection of .ogg extension."""
        assert get_extension_from_url("https://example.com/audio.ogg") == ".ogg"

    def test_flac_extension(self):
        """Test detection of .flac extension."""
        assert get_extension_from_url("https://example.com/audio.flac") == ".flac"

    def test_unknown_extension_defaults_to_mp3(self):
        """Test that unknown extension defaults to .mp3."""
        assert get_extension_from_url("https://example.com/audio") == ".mp3"
        assert get_extension_from_url("https://example.com/audio.xyz") == ".mp3"

    def test_extension_in_path(self):
        """Test detection of extension in URL path."""
        url = "https://static.inaturalist.org/sounds/12345.wav?download=true"
        assert get_extension_from_url(url) == ".wav"


class TestGetExtensionFromContentType:
    """Tests for get_extension_from_content_type function."""

    def test_audio_mpeg(self):
        """Test audio/mpeg content type."""
        assert get_extension_from_content_type("audio/mpeg") == ".mp3"

    def test_audio_wav(self):
        """Test audio/wav content types."""
        assert get_extension_from_content_type("audio/wav") == ".wav"
        assert get_extension_from_content_type("audio/x-wav") == ".wav"
        assert get_extension_from_content_type("audio/wave") == ".wav"

    def test_audio_m4a(self):
        """Test audio/m4a content types."""
        assert get_extension_from_content_type("audio/m4a") == ".m4a"
        assert get_extension_from_content_type("audio/x-m4a") == ".m4a"
        assert get_extension_from_content_type("audio/mp4") == ".m4a"

    def test_audio_ogg(self):
        """Test audio/ogg content type."""
        assert get_extension_from_content_type("audio/ogg") == ".ogg"

    def test_audio_flac(self):
        """Test audio/flac content types."""
        assert get_extension_from_content_type("audio/flac") == ".flac"
        assert get_extension_from_content_type("audio/x-flac") == ".flac"

    def test_content_type_with_charset(self):
        """Test content type with charset parameter."""
        assert get_extension_from_content_type("audio/mpeg; charset=utf-8") == ".mp3"

    def test_unknown_content_type_returns_empty(self):
        """Test that unknown content type returns empty string."""
        assert get_extension_from_content_type("application/octet-stream") == ""
        assert get_extension_from_content_type("text/html") == ""


class TestFindSpeciesName:
    """Tests for find_species_name function."""

    def test_subspecies_finds_species(self):
        """Test finding species from subspecies name."""
        all_categories = {
            "Lithobates sphenocephalus",
            "Lithobates sphenocephalus utricularius",
            "Lithobates catesbeianus"
        }

        result = find_species_name(
            "Lithobates sphenocephalus utricularius",
            all_categories
        )

        assert result == "Lithobates sphenocephalus"

    def test_species_returns_itself(self):
        """Test that species with no subspecies returns itself."""
        all_categories = {
            "Lithobates catesbeianus",
            "Rana temporaria"
        }

        result = find_species_name("Lithobates catesbeianus", all_categories)
        assert result == "Lithobates catesbeianus"

    def test_empty_category_returns_empty(self):
        """Test that empty category returns empty string."""
        result = find_species_name("", {"species_a", "species_b"})
        assert result == ""

    def test_no_match_returns_original(self):
        """Test that no match returns original category."""
        all_categories = {"species_a", "species_b"}
        result = find_species_name("species_c", all_categories)
        assert result == "species_c"


class TestEnsureDirectory:
    """Tests for ensure_directory function."""

    def test_creates_directory(self, temp_dir):
        """Test that directory is created."""
        new_dir = temp_dir / "new_subdir"
        assert not new_dir.exists()

        result = ensure_directory(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir

    def test_creates_nested_directories(self, temp_dir):
        """Test that nested directories are created."""
        nested = temp_dir / "level1" / "level2" / "level3"
        assert not nested.exists()

        ensure_directory(nested)

        assert nested.exists()

    def test_existing_directory_ok(self, temp_dir):
        """Test that existing directory doesn't raise error."""
        existing = temp_dir / "existing"
        existing.mkdir()

        # Should not raise
        result = ensure_directory(existing)
        assert result == existing


class TestGetRelativePath:
    """Tests for get_relative_path function."""

    def test_relative_path_calculation(self, temp_dir):
        """Test relative path calculation."""
        base = temp_dir
        file = temp_dir / "subdir" / "file.txt"

        result = get_relative_path(file, base)
        assert result == "subdir/file.txt"

    def test_file_not_under_base_returns_filename(self, temp_dir):
        """Test that file not under base returns filename only."""
        from pathlib import Path

        base = temp_dir / "base"
        file = Path("/other/path/file.txt")

        result = get_relative_path(file, base)
        assert result == "file.txt"
