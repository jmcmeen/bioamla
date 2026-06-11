"""System dependency checking and installation.

Folds the former ``services/dependency.py`` into plain raising functions and
dataclasses. Checks for system-level tools required by bioamla (FFmpeg,
libsndfile, PortAudio) and can attempt to install them via the platform package
manager.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

from bioamla.exceptions import ProcessingError


@dataclass
class DependencyInfo:
    """Information about a system dependency."""

    name: str
    description: str
    required_for: str
    installed: bool
    version: str | None = None
    install_hint: str | None = None


@dataclass
class DependencyReport:
    """Report of all dependency statuses."""

    os_type: str
    all_installed: bool
    dependencies: list[DependencyInfo] = field(default_factory=list)
    install_command: str | None = None


# =============================================================================
# OS and package-manager detection
# =============================================================================


def detect_os() -> str:
    """Detect the operating system / package-manager family.

    Returns:
        One of: 'debian', 'fedora', 'rhel', 'arch', 'macos', 'windows',
        'unknown-linux', 'unknown'.
    """
    system = platform.system().lower()

    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    if system == "linux":
        if shutil.which("apt-get"):
            return "debian"
        if shutil.which("dnf"):
            return "fedora"
        if shutil.which("yum"):
            return "rhel"
        if shutil.which("pacman"):
            return "arch"
        return "unknown-linux"
    return "unknown"


# =============================================================================
# Individual dependency checks
# =============================================================================


def check_ffmpeg() -> DependencyInfo:
    """Check if FFmpeg is installed."""
    version = None
    installed = False

    if shutil.which("ffmpeg"):
        installed = True
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5
            )
            first_line = result.stdout.split("\n")[0]
            if "version" in first_line.lower():
                version = first_line.split("version")[1].split()[0].strip()
        except Exception:
            pass

    return DependencyInfo(
        name="FFmpeg",
        description="Audio format conversion",
        required_for="MP3, FLAC, and other formats (WAV works without)",
        installed=installed,
        version=version,
    )


def check_libsndfile() -> DependencyInfo:
    """Check if libsndfile is installed."""
    installed = False
    version = None

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

    if not installed:
        try:
            import soundfile

            installed = True
            version = getattr(soundfile, "__version__", "available")
        except (ImportError, OSError):
            pass

    return DependencyInfo(
        name="libsndfile",
        description="Audio file I/O library",
        required_for="Reading/writing WAV, FLAC, OGG files",
        installed=installed,
        version=version,
    )


def check_portaudio() -> DependencyInfo:
    """Check if PortAudio is installed."""
    installed = False
    version = None

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

    if not installed:
        try:
            import sounddevice

            installed = True
            if sounddevice.query_hostapis():
                version = "available"
        except (ImportError, OSError):
            pass

    return DependencyInfo(
        name="PortAudio",
        description="Audio hardware access",
        required_for="Real-time recording (bioamla realtime commands)",
        installed=installed,
        version=version,
    )


# =============================================================================
# Install commands
# =============================================================================


def get_install_commands(os_type: str) -> dict[str, str]:
    """Get per-dependency install commands for an OS family."""
    commands: dict[str, dict[str, str]] = {
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


def get_full_install_command(os_type: str) -> str | None:
    """Get a single command to install all dependencies for an OS family."""
    commands = {
        "debian": "sudo apt install ffmpeg libsndfile1 portaudio19-dev",
        "fedora": "sudo dnf install ffmpeg libsndfile portaudio",
        "rhel": "sudo yum install ffmpeg libsndfile portaudio",
        "arch": "sudo pacman -S ffmpeg libsndfile portaudio",
        "macos": "brew install ffmpeg libsndfile portaudio",
    }
    return commands.get(os_type)


# =============================================================================
# Batch check
# =============================================================================


def check_all() -> DependencyReport:
    """Check all system dependencies and return an aggregated report."""
    os_type = detect_os()
    deps = [check_ffmpeg(), check_libsndfile(), check_portaudio()]

    install_commands = get_install_commands(os_type)
    for dep in deps:
        dep.install_hint = install_commands.get(dep.name.lower())

    all_installed = all(d.installed for d in deps)
    return DependencyReport(
        os_type=os_type,
        all_installed=all_installed,
        dependencies=deps,
        install_command=get_full_install_command(os_type),
    )


# =============================================================================
# Installation
# =============================================================================


def _has_sudo() -> bool:
    """Check whether passwordless sudo is available."""
    if sys.platform == "win32":
        return False
    try:
        result = subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def run_install(os_type: str | None = None) -> tuple[bool, str]:
    """Attempt to install system dependencies via the platform package manager.

    Returns a ``(success, message)`` tuple. Use :func:`install` for a raising
    variant.
    """
    if os_type is None:
        os_type = detect_os()

    if os_type == "windows":
        return False, (
            "Automatic installation not supported on Windows.\n"
            "Please install manually:\n"
            "  - FFmpeg: https://ffmpeg.org/download.html or 'choco install ffmpeg'\n"
            "  - libsndfile/PortAudio are usually bundled with pip packages"
        )

    if os_type in ("unknown", "unknown-linux"):
        return False, (
            "Could not detect your operating system.\n"
            "Please install manually:\n"
            "  - FFmpeg (for audio format support via pydub)\n"
            "  - libsndfile (for audio I/O)\n"
            "  - PortAudio (for real-time recording)"
        )

    command = get_full_install_command(os_type)
    if not command:
        return False, f"No install command available for {os_type}"

    if os_type != "macos" and not _has_sudo():
        return False, f"Installation requires sudo privileges.\nPlease run: {command}"

    try:
        if os_type == "macos":
            result = subprocess.run(
                ["brew", "install", "ffmpeg", "libsndfile", "portaudio"],
                capture_output=True,
                text=True,
                timeout=300,
            )
        else:
            result = subprocess.run(command.split(), capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            return True, "Dependencies installed successfully!"
        return False, f"Installation failed:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "Installation timed out. Please run manually:\n" + command
    except FileNotFoundError as e:
        return False, f"Package manager not found: {e}\nPlease run manually:\n{command}"
    except Exception as e:
        return False, f"Installation failed: {e}\nPlease run manually:\n{command}"


def install(os_type: str | None = None) -> str:
    """Install system dependencies, raising on failure.

    Returns:
        Success message string.

    Raises:
        ProcessingError: If installation fails or is unsupported.
    """
    success, message = run_install(os_type)
    if not success:
        raise ProcessingError(message)
    return message


__all__ = [
    "DependencyInfo",
    "DependencyReport",
    "detect_os",
    "check_ffmpeg",
    "check_libsndfile",
    "check_portaudio",
    "check_all",
    "get_install_commands",
    "get_full_install_command",
    "run_install",
    "install",
]
