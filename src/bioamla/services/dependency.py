# services/dependency.py
"""
Service for system dependency checking and installation.

This module provides utilities to check for and install system-level
dependencies required by bioamla (FFmpeg, PortAudio, libsndfile).
"""

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from bioamla.models.dependency import DependencyInfo, DependencyReport

from .base import BaseService, ServiceResult


# =============================================================================
# Core Dependency Data Structures
# =============================================================================


@dataclass
class DependencyStatus:
    """Status of a system dependency."""

    name: str
    description: str
    required_for: str
    installed: bool
    version: Optional[str] = None
    install_hint: Optional[str] = None


# =============================================================================
# OS and Package Manager Detection
# =============================================================================


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


# =============================================================================
# Individual Dependency Checks
# =============================================================================


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

    # Fallback: try importing soundfile
    if not installed:
        try:
            import soundfile

            installed = True
            version = getattr(soundfile, "__version__", "available")
        except (ImportError, OSError):
            pass

    return DependencyStatus(
        name="libsndfile",
        description="Audio file I/O library",
        required_for="Reading/writing WAV, FLAC, OGG files",
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


# =============================================================================
# Batch Dependency Checks
# =============================================================================


def check_all_dependencies() -> List[DependencyStatus]:
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


# =============================================================================
# Installation Commands
# =============================================================================


def get_install_commands(os_type: str) -> Dict[str, str]:
    """Get install commands for each dependency based on OS.

    Args:
        os_type: The detected OS type.

    Returns:
        Dict mapping dependency name to install command.
    """
    commands: Dict[str, Dict[str, str]] = {
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


# =============================================================================
# Installation
# =============================================================================


def run_install(os_type: Optional[str] = None) -> Tuple[bool, str]:
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
            "  - libsndfile/PortAudio are usually bundled with pip packages"
        )

    if os_type == "unknown" or os_type == "unknown-linux":
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

    # Check for required privileges on macOS (Homebrew doesn't need sudo)
    if os_type != "macos" and not _has_sudo():
        return False, (f"Installation requires sudo privileges.\nPlease run: {command}")

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


# =============================================================================
# Service Class
# =============================================================================


class DependencyService(BaseService):
    """
    Service for system dependency checking and installation.

    Provides ServiceResult-wrapped methods for dependency management.
    """

    def detect_os(self) -> ServiceResult[str]:
        """
        Detect the operating system and package manager.

        Returns:
            ServiceResult containing OS type string
        """
        try:
            os_type = detect_os()

            return ServiceResult.ok(
                data=os_type,
                message=f"Detected OS: {os_type}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to detect OS: {e}")

    def check_ffmpeg(self) -> ServiceResult[DependencyInfo]:
        """
        Check if FFmpeg is installed.

        Returns:
            ServiceResult containing DependencyInfo for FFmpeg
        """
        try:
            status = check_ffmpeg()

            info = DependencyInfo(
                name=status.name,
                description=status.description,
                required_for=status.required_for,
                installed=status.installed,
                version=status.version,
                install_hint=status.install_hint,
            )

            return ServiceResult.ok(
                data=info,
                message=f"FFmpeg: {'installed' if info.installed else 'not installed'}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to check FFmpeg: {e}")

    def check_libsndfile(self) -> ServiceResult[DependencyInfo]:
        """
        Check if libsndfile is installed.

        Returns:
            ServiceResult containing DependencyInfo for libsndfile
        """
        try:
            status = check_libsndfile()

            info = DependencyInfo(
                name=status.name,
                description=status.description,
                required_for=status.required_for,
                installed=status.installed,
                version=status.version,
                install_hint=status.install_hint,
            )

            return ServiceResult.ok(
                data=info,
                message=f"libsndfile: {'installed' if info.installed else 'not installed'}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to check libsndfile: {e}")

    def check_portaudio(self) -> ServiceResult[DependencyInfo]:
        """
        Check if PortAudio is installed.

        Returns:
            ServiceResult containing DependencyInfo for PortAudio
        """
        try:
            status = check_portaudio()

            info = DependencyInfo(
                name=status.name,
                description=status.description,
                required_for=status.required_for,
                installed=status.installed,
                version=status.version,
                install_hint=status.install_hint,
            )

            return ServiceResult.ok(
                data=info,
                message=f"PortAudio: {'installed' if info.installed else 'not installed'}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to check PortAudio: {e}")

    def check_all(self) -> ServiceResult[DependencyReport]:
        """
        Check all system dependencies.

        Returns:
            ServiceResult containing DependencyReport
        """
        try:
            os_type = detect_os()
            statuses = check_all_dependencies()

            dependencies = [
                DependencyInfo(
                    name=s.name,
                    description=s.description,
                    required_for=s.required_for,
                    installed=s.installed,
                    version=s.version,
                    install_hint=s.install_hint,
                )
                for s in statuses
            ]

            all_installed = all(d.installed for d in dependencies)
            install_command = get_full_install_command(os_type)

            report = DependencyReport(
                os_type=os_type,
                all_installed=all_installed,
                dependencies=dependencies,
                install_command=install_command,
            )

            if all_installed:
                message = "All dependencies installed"
            else:
                missing = [d.name for d in dependencies if not d.installed]
                message = f"Missing dependencies: {', '.join(missing)}"

            return ServiceResult.ok(
                data=report,
                message=message,
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to check dependencies: {e}")

    def get_install_command(
        self,
        os_type: Optional[str] = None,
    ) -> ServiceResult[Optional[str]]:
        """
        Get the command to install all dependencies.

        Args:
            os_type: Override OS detection. If None, auto-detect.

        Returns:
            ServiceResult containing the install command or None
        """
        try:
            if os_type is None:
                os_type = detect_os()

            command = get_full_install_command(os_type)

            if command:
                return ServiceResult.ok(
                    data=command,
                    message=f"Install command for {os_type}",
                )
            else:
                return ServiceResult.ok(
                    data=None,
                    message=f"No install command available for {os_type}",
                )
        except Exception as e:
            return ServiceResult.fail(f"Failed to get install command: {e}")

    def install(
        self,
        os_type: Optional[str] = None,
    ) -> ServiceResult[bool]:
        """
        Install system dependencies.

        Args:
            os_type: Override OS detection. If None, auto-detect.

        Returns:
            ServiceResult containing True on success
        """
        try:
            success, message = run_install(os_type)

            if success:
                return ServiceResult.ok(
                    data=True,
                    message=message,
                )
            else:
                return ServiceResult.fail(message)
        except Exception as e:
            return ServiceResult.fail(f"Failed to install dependencies: {e}")
