"""
System dependency checking and installation for bioamla.

This module provides utilities to check for and install system-level
dependencies required by bioamla (FFmpeg, libsndfile, PortAudio).
"""

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class DependencyStatus:
    """Status of a system dependency."""

    name: str
    description: str
    required_for: str
    installed: bool
    version: Optional[str] = None
    install_hint: Optional[str] = None


def detect_os() -> str:
    """Detect the operating system and package manager.

    Returns:
        One of: 'debian', 'fedora', 'rhel', 'arch', 'macos', 'windows', 'unknown'
    """
    system = platform.system().lower()

    if system == "darwin":
        return "macos"
    elif system == "windows":
        return "windows"
    elif system == "linux":
        # Check for package managers to identify distro family
        if shutil.which("apt-get"):
            return "debian"
        elif shutil.which("dnf"):
            return "fedora"
        elif shutil.which("yum"):
            return "rhel"
        elif shutil.which("pacman"):
            return "arch"
        else:
            return "unknown-linux"
    else:
        return "unknown"


def check_ffmpeg() -> DependencyStatus:
    """Check if FFmpeg is installed."""
    version = None
    installed = False

    if shutil.which("ffmpeg"):
        installed = True
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # Extract version from first line
            first_line = result.stdout.split("\n")[0]
            if "version" in first_line.lower():
                version = first_line.split("version")[1].split()[0].strip()
        except Exception:
            pass

    return DependencyStatus(
        name="FFmpeg",
        description="Audio format conversion",
        required_for="MP3, FLAC, and other formats (WAV works without)",
        installed=installed,
        version=version,
    )


def check_libsndfile() -> DependencyStatus:
    """Check if libsndfile is installed."""
    installed = False
    version = None

    # Try pkg-config first
    if shutil.which("pkg-config"):
        try:
            result = subprocess.run(
                ["pkg-config", "--modversion", "sndfile"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                installed = True
                version = result.stdout.strip()
        except Exception:
            pass

    # Fallback: try importing soundfile (will fail if libsndfile missing)
    if not installed:
        try:
            import soundfile

            installed = True
            version = getattr(soundfile, "__libsndfile_version__", "unknown")
        except (ImportError, OSError):
            pass

    return DependencyStatus(
        name="libsndfile",
        description="Audio file I/O library",
        required_for="Reading/writing audio files",
        installed=installed,
        version=version,
    )


def check_portaudio() -> DependencyStatus:
    """Check if PortAudio is installed."""
    installed = False
    version = None

    # Try pkg-config first
    if shutil.which("pkg-config"):
        try:
            result = subprocess.run(
                ["pkg-config", "--modversion", "portaudio-2.0"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                installed = True
                version = result.stdout.strip()
        except Exception:
            pass

    # Fallback: try importing sounddevice
    if not installed:
        try:
            import sounddevice

            installed = True
            pa_version = sounddevice.query_hostapis()
            if pa_version:
                version = "available"
        except (ImportError, OSError):
            pass

    return DependencyStatus(
        name="PortAudio",
        description="Audio hardware access",
        required_for="Real-time recording (bioamla realtime commands)",
        installed=installed,
        version=version,
    )


def check_all_dependencies() -> list[DependencyStatus]:
    """Check all system dependencies.

    Returns:
        List of DependencyStatus objects for each dependency.
    """
    os_type = detect_os()

    deps = [
        check_ffmpeg(),
        check_libsndfile(),
        check_portaudio(),
    ]

    # Add install hints based on OS
    install_commands = get_install_commands(os_type)
    for dep in deps:
        dep.install_hint = install_commands.get(dep.name.lower())

    return deps


def get_install_commands(os_type: str) -> dict[str, str]:
    """Get install commands for each dependency based on OS.

    Args:
        os_type: The detected OS type.

    Returns:
        Dict mapping dependency name to install command.
    """
    commands = {
        "debian": {
            "ffmpeg": "sudo apt install ffmpeg",
            "libsndfile": "sudo apt install libsndfile1",
            "portaudio": "sudo apt install portaudio19-dev",
        },
        "fedora": {
            "ffmpeg": "sudo dnf install ffmpeg",
            "libsndfile": "sudo dnf install libsndfile",
            "portaudio": "sudo dnf install portaudio",
        },
        "rhel": {
            "ffmpeg": "sudo yum install ffmpeg",
            "libsndfile": "sudo yum install libsndfile",
            "portaudio": "sudo yum install portaudio",
        },
        "arch": {
            "ffmpeg": "sudo pacman -S ffmpeg",
            "libsndfile": "sudo pacman -S libsndfile",
            "portaudio": "sudo pacman -S portaudio",
        },
        "macos": {
            "ffmpeg": "brew install ffmpeg",
            "libsndfile": "brew install libsndfile",
            "portaudio": "brew install portaudio",
        },
        "windows": {
            "ffmpeg": "choco install ffmpeg  # or download from ffmpeg.org",
            "libsndfile": "pip install soundfile  # usually bundled on Windows",
            "portaudio": "pip install sounddevice  # usually bundled on Windows",
        },
    }

    return commands.get(os_type, {})


def get_full_install_command(os_type: str) -> Optional[str]:
    """Get a single command to install all dependencies.

    Args:
        os_type: The detected OS type.

    Returns:
        Full install command string, or None if not available.
    """
    commands = {
        "debian": "sudo apt install ffmpeg libsndfile1 portaudio19-dev",
        "fedora": "sudo dnf install ffmpeg libsndfile portaudio",
        "rhel": "sudo yum install ffmpeg libsndfile portaudio",
        "arch": "sudo pacman -S ffmpeg libsndfile portaudio",
        "macos": "brew install ffmpeg libsndfile portaudio",
    }
    return commands.get(os_type)


def run_install(os_type: Optional[str] = None) -> tuple[bool, str]:
    """Run the system dependency installation.

    Args:
        os_type: Override OS detection. If None, auto-detect.

    Returns:
        Tuple of (success, message).
    """
    if os_type is None:
        os_type = detect_os()

    if os_type == "windows":
        return False, (
            "Automatic installation not supported on Windows.\n"
            "Please install manually:\n"
            "  - FFmpeg: https://ffmpeg.org/download.html or 'choco install ffmpeg'\n"
            "  - libsndfile and PortAudio are usually bundled with pip packages"
        )

    if os_type == "unknown" or os_type == "unknown-linux":
        return False, (
            "Could not detect your operating system.\n"
            "Please install manually:\n"
            "  - FFmpeg (for audio format support)\n"
            "  - libsndfile (for audio file I/O)\n"
            "  - PortAudio (for real-time recording)"
        )

    command = get_full_install_command(os_type)
    if not command:
        return False, f"No install command available for {os_type}"

    # Check for required privileges on macOS (Homebrew doesn't need sudo)
    if os_type != "macos" and not _has_sudo():
        return False, (
            f"Installation requires sudo privileges.\n"
            f"Please run: {command}"
        )

    try:
        # Split command for subprocess
        if os_type == "macos":
            # Homebrew command
            result = subprocess.run(
                ["brew", "install", "ffmpeg", "libsndfile", "portaudio"],
                capture_output=True,
                text=True,
                timeout=300,
            )
        else:
            # Linux with sudo
            parts = command.split()
            result = subprocess.run(
                parts,
                capture_output=True,
                text=True,
                timeout=300,
            )

        if result.returncode == 0:
            return True, "Dependencies installed successfully!"
        else:
            return False, f"Installation failed:\n{result.stderr}"

    except subprocess.TimeoutExpired:
        return False, "Installation timed out. Please run manually:\n" + command
    except FileNotFoundError as e:
        return False, f"Package manager not found: {e}\nPlease run manually:\n{command}"
    except Exception as e:
        return False, f"Installation failed: {e}\nPlease run manually:\n{command}"


def _has_sudo() -> bool:
    """Check if we can run sudo."""
    if sys.platform == "win32":
        return False

    try:
        result = subprocess.run(
            ["sudo", "-n", "true"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False