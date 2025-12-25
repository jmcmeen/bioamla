# services/dependency.py
"""
Service for system dependency checking and installation.
"""

from dataclasses import dataclass
from typing import List, Optional

from .base import BaseService, ServiceResult, ToDictMixin


@dataclass
class DependencyInfo(ToDictMixin):
    """Information about a system dependency."""

    name: str
    description: str
    required_for: str
    installed: bool
    version: Optional[str] = None
    install_hint: Optional[str] = None


@dataclass
class DependencyReport(ToDictMixin):
    """Report of all dependency statuses."""

    os_type: str
    all_installed: bool
    dependencies: List[DependencyInfo]
    install_command: Optional[str] = None


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
            from bioamla.core.deps import detect_os as core_detect_os

            os_type = core_detect_os()

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
            from bioamla.core.deps import check_ffmpeg as core_check

            status = core_check()

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
            from bioamla.core.deps import check_libsndfile as core_check

            status = core_check()

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
            from bioamla.core.deps import check_portaudio as core_check

            status = core_check()

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
            from bioamla.core.deps import (
                check_all_dependencies,
                detect_os,
                get_full_install_command,
            )

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
            from bioamla.core.deps import detect_os, get_full_install_command

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
            from bioamla.core.deps import run_install

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
