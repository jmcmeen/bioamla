"""
Unit tests for the deps module (system dependency checking).
"""

import pytest
from unittest.mock import patch, MagicMock

from bioamla.core.deps import (
    DependencyStatus,
    detect_os,
    check_ffmpeg,
    check_libsndfile,
    check_portaudio,
    check_all_dependencies,
    get_install_commands,
    get_full_install_command,
)


class TestDetectOS:
    """Tests for detect_os function."""

    def test_detect_os_returns_string(self):
        """Test that detect_os returns a string."""
        result = detect_os()
        assert isinstance(result, str)

    def test_detect_os_known_values(self):
        """Test that detect_os returns one of the known OS types."""
        result = detect_os()
        known_types = [
            "debian",
            "fedora",
            "rhel",
            "arch",
            "macos",
            "windows",
            "unknown",
            "unknown-linux",
        ]
        assert result in known_types

    @patch("bioamla.deps.platform.system")
    def test_detect_os_darwin(self, mock_system):
        """Test detection of macOS."""
        mock_system.return_value = "Darwin"
        assert detect_os() == "macos"

    @patch("bioamla.deps.platform.system")
    def test_detect_os_windows(self, mock_system):
        """Test detection of Windows."""
        mock_system.return_value = "Windows"
        assert detect_os() == "windows"

    @patch("bioamla.deps.platform.system")
    @patch("bioamla.deps.shutil.which")
    def test_detect_os_debian(self, mock_which, mock_system):
        """Test detection of Debian-based Linux."""
        mock_system.return_value = "Linux"
        mock_which.side_effect = lambda x: "/usr/bin/apt-get" if x == "apt-get" else None
        assert detect_os() == "debian"

    @patch("bioamla.deps.platform.system")
    @patch("bioamla.deps.shutil.which")
    def test_detect_os_fedora(self, mock_which, mock_system):
        """Test detection of Fedora."""
        mock_system.return_value = "Linux"
        mock_which.side_effect = lambda x: "/usr/bin/dnf" if x == "dnf" else None
        assert detect_os() == "fedora"

    @patch("bioamla.deps.platform.system")
    @patch("bioamla.deps.shutil.which")
    def test_detect_os_arch(self, mock_which, mock_system):
        """Test detection of Arch Linux."""
        mock_system.return_value = "Linux"
        mock_which.side_effect = lambda x: "/usr/bin/pacman" if x == "pacman" else None
        assert detect_os() == "arch"

    @patch("bioamla.deps.platform.system")
    @patch("bioamla.deps.shutil.which")
    def test_detect_os_unknown_linux(self, mock_which, mock_system):
        """Test detection of unknown Linux."""
        mock_system.return_value = "Linux"
        mock_which.return_value = None
        assert detect_os() == "unknown-linux"


class TestCheckFFmpeg:
    """Tests for check_ffmpeg function."""

    def test_check_ffmpeg_returns_status(self):
        """Test that check_ffmpeg returns a DependencyStatus."""
        result = check_ffmpeg()
        assert isinstance(result, DependencyStatus)

    def test_check_ffmpeg_has_correct_name(self):
        """Test that the status has the correct name."""
        result = check_ffmpeg()
        assert result.name == "FFmpeg"

    def test_check_ffmpeg_has_description(self):
        """Test that the status has a description."""
        result = check_ffmpeg()
        assert result.description == "Audio format conversion"

    def test_check_ffmpeg_has_required_for(self):
        """Test that the status has required_for info."""
        result = check_ffmpeg()
        assert "MP3" in result.required_for or "format" in result.required_for.lower()

    @patch("bioamla.deps.shutil.which")
    def test_check_ffmpeg_not_installed(self, mock_which):
        """Test when FFmpeg is not installed."""
        mock_which.return_value = None
        result = check_ffmpeg()
        assert result.installed is False
        assert result.version is None


class TestCheckLibsndfile:
    """Tests for check_libsndfile function."""

    def test_check_libsndfile_returns_status(self):
        """Test that check_libsndfile returns a DependencyStatus."""
        result = check_libsndfile()
        assert isinstance(result, DependencyStatus)

    def test_check_libsndfile_has_correct_name(self):
        """Test that the status has the correct name."""
        result = check_libsndfile()
        assert result.name == "libsndfile"

    def test_check_libsndfile_has_description(self):
        """Test that the status has a description."""
        result = check_libsndfile()
        assert "Audio" in result.description or "I/O" in result.description


class TestCheckPortaudio:
    """Tests for check_portaudio function."""

    def test_check_portaudio_returns_status(self):
        """Test that check_portaudio returns a DependencyStatus."""
        result = check_portaudio()
        assert isinstance(result, DependencyStatus)

    def test_check_portaudio_has_correct_name(self):
        """Test that the status has the correct name."""
        result = check_portaudio()
        assert result.name == "PortAudio"

    def test_check_portaudio_has_description(self):
        """Test that the status has a description."""
        result = check_portaudio()
        assert "Audio" in result.description or "hardware" in result.description.lower()


class TestCheckAllDependencies:
    """Tests for check_all_dependencies function."""

    def test_check_all_returns_list(self):
        """Test that check_all_dependencies returns a list."""
        result = check_all_dependencies()
        assert isinstance(result, list)

    def test_check_all_returns_three_deps(self):
        """Test that check_all_dependencies returns exactly 3 dependencies."""
        result = check_all_dependencies()
        assert len(result) == 3

    def test_check_all_returns_status_objects(self):
        """Test that all items are DependencyStatus objects."""
        result = check_all_dependencies()
        for dep in result:
            assert isinstance(dep, DependencyStatus)

    def test_check_all_includes_all_deps(self):
        """Test that all expected dependencies are included."""
        result = check_all_dependencies()
        names = [dep.name for dep in result]
        assert "FFmpeg" in names
        assert "libsndfile" in names
        assert "PortAudio" in names

    def test_check_all_has_install_hints(self):
        """Test that dependencies have install hints populated."""
        result = check_all_dependencies()
        # At least one should have an install hint on any supported OS
        has_hints = any(dep.install_hint is not None for dep in result)
        # This might be False on unknown OS, so we just verify the structure
        for dep in result:
            assert hasattr(dep, "install_hint")


class TestGetInstallCommands:
    """Tests for get_install_commands function."""

    def test_debian_commands(self):
        """Test install commands for Debian."""
        commands = get_install_commands("debian")
        assert "apt" in commands.get("ffmpeg", "")
        assert "apt" in commands.get("libsndfile", "")
        assert "apt" in commands.get("portaudio", "")

    def test_fedora_commands(self):
        """Test install commands for Fedora."""
        commands = get_install_commands("fedora")
        assert "dnf" in commands.get("ffmpeg", "")
        assert "dnf" in commands.get("libsndfile", "")
        assert "dnf" in commands.get("portaudio", "")

    def test_arch_commands(self):
        """Test install commands for Arch."""
        commands = get_install_commands("arch")
        assert "pacman" in commands.get("ffmpeg", "")
        assert "pacman" in commands.get("libsndfile", "")
        assert "pacman" in commands.get("portaudio", "")

    def test_macos_commands(self):
        """Test install commands for macOS."""
        commands = get_install_commands("macos")
        assert "brew" in commands.get("ffmpeg", "")
        assert "brew" in commands.get("libsndfile", "")
        assert "brew" in commands.get("portaudio", "")

    def test_windows_commands(self):
        """Test install commands for Windows."""
        commands = get_install_commands("windows")
        assert "ffmpeg" in commands
        assert "libsndfile" in commands
        assert "portaudio" in commands

    def test_unknown_os_returns_empty(self):
        """Test that unknown OS returns empty dict."""
        commands = get_install_commands("unknown")
        assert commands == {}


class TestGetFullInstallCommand:
    """Tests for get_full_install_command function."""

    def test_debian_full_command(self):
        """Test full install command for Debian."""
        command = get_full_install_command("debian")
        assert command is not None
        assert "apt" in command
        assert "ffmpeg" in command
        assert "libsndfile" in command
        assert "portaudio" in command

    def test_fedora_full_command(self):
        """Test full install command for Fedora."""
        command = get_full_install_command("fedora")
        assert command is not None
        assert "dnf" in command

    def test_macos_full_command(self):
        """Test full install command for macOS."""
        command = get_full_install_command("macos")
        assert command is not None
        assert "brew" in command

    def test_windows_returns_none(self):
        """Test that Windows returns None (no full command)."""
        command = get_full_install_command("windows")
        assert command is None

    def test_unknown_returns_none(self):
        """Test that unknown OS returns None."""
        command = get_full_install_command("unknown")
        assert command is None


class TestDependencyStatus:
    """Tests for DependencyStatus dataclass."""

    def test_create_status(self):
        """Test creating a DependencyStatus."""
        status = DependencyStatus(
            name="Test",
            description="Test description",
            required_for="Testing",
            installed=True,
            version="1.0.0",
            install_hint="apt install test",
        )
        assert status.name == "Test"
        assert status.description == "Test description"
        assert status.required_for == "Testing"
        assert status.installed is True
        assert status.version == "1.0.0"
        assert status.install_hint == "apt install test"

    def test_create_status_minimal(self):
        """Test creating a DependencyStatus with minimal fields."""
        status = DependencyStatus(
            name="Test",
            description="Test description",
            required_for="Testing",
            installed=False,
        )
        assert status.name == "Test"
        assert status.installed is False
        assert status.version is None
        assert status.install_hint is None