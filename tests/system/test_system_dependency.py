"""Tests for bioamla.system.dependency (deterministic via mocking)."""

from __future__ import annotations

import subprocess

import pytest

from bioamla.exceptions import ProcessingError
from bioamla.system import dependency as dep
from bioamla.system.dependency import (
    DependencyInfo,
    DependencyReport,
    check_all,
    check_ffmpeg,
    check_libsndfile,
    check_portaudio,
    detect_os,
    get_full_install_command,
    get_install_commands,
    install,
    run_install,
)


# ----------------------------------------------------------------------------
# detect_os
# ----------------------------------------------------------------------------
def test_detect_os_macos(monkeypatch):
    monkeypatch.setattr(dep.platform, "system", lambda: "Darwin")
    assert detect_os() == "macos"


def test_detect_os_windows(monkeypatch):
    monkeypatch.setattr(dep.platform, "system", lambda: "Windows")
    assert detect_os() == "windows"


@pytest.mark.parametrize(
    "available,expected",
    [
        ("apt-get", "debian"),
        ("dnf", "fedora"),
        ("yum", "rhel"),
        ("pacman", "arch"),
    ],
)
def test_detect_os_linux_families(monkeypatch, available, expected):
    monkeypatch.setattr(dep.platform, "system", lambda: "Linux")
    monkeypatch.setattr(dep.shutil, "which", lambda tool: tool if tool == available else None)
    assert detect_os() == expected


def test_detect_os_unknown_linux(monkeypatch):
    monkeypatch.setattr(dep.platform, "system", lambda: "Linux")
    monkeypatch.setattr(dep.shutil, "which", lambda tool: None)
    assert detect_os() == "unknown-linux"


def test_detect_os_unknown(monkeypatch):
    monkeypatch.setattr(dep.platform, "system", lambda: "Solaris")
    assert detect_os() == "unknown"


# ----------------------------------------------------------------------------
# check_ffmpeg
# ----------------------------------------------------------------------------
def test_check_ffmpeg_present(monkeypatch):
    monkeypatch.setattr(dep.shutil, "which", lambda t: "/usr/bin/ffmpeg")

    def fake_run(*a, **k):
        return subprocess.CompletedProcess(a, 0, stdout="ffmpeg version 6.1 extra\n", stderr="")

    monkeypatch.setattr(dep.subprocess, "run", fake_run)
    info = check_ffmpeg()
    assert isinstance(info, DependencyInfo)
    assert info.installed is True
    assert info.version == "6.1"


def test_check_ffmpeg_present_run_raises(monkeypatch):
    monkeypatch.setattr(dep.shutil, "which", lambda t: "/usr/bin/ffmpeg")

    def fake_run(*a, **k):
        raise OSError("boom")

    monkeypatch.setattr(dep.subprocess, "run", fake_run)
    info = check_ffmpeg()
    assert info.installed is True
    assert info.version is None


def test_check_ffmpeg_missing(monkeypatch):
    monkeypatch.setattr(dep.shutil, "which", lambda t: None)
    info = check_ffmpeg()
    assert info.installed is False


# ----------------------------------------------------------------------------
# check_libsndfile
# ----------------------------------------------------------------------------
def test_check_libsndfile_via_pkgconfig(monkeypatch):
    monkeypatch.setattr(dep.shutil, "which", lambda t: "/usr/bin/pkg-config")

    def fake_run(*a, **k):
        return subprocess.CompletedProcess(a, 0, stdout="1.2.0\n", stderr="")

    monkeypatch.setattr(dep.subprocess, "run", fake_run)
    info = check_libsndfile()
    assert info.installed is True
    assert info.version == "1.2.0"


def test_check_libsndfile_via_soundfile(monkeypatch):
    # pkg-config absent -> falls back to importing soundfile
    monkeypatch.setattr(dep.shutil, "which", lambda t: None)
    info = check_libsndfile()
    # soundfile is part of the runtime stack; should be importable
    assert info.installed is True


# ----------------------------------------------------------------------------
# check_portaudio
# ----------------------------------------------------------------------------
def test_check_portaudio_via_pkgconfig(monkeypatch):
    monkeypatch.setattr(dep.shutil, "which", lambda t: "/usr/bin/pkg-config")

    def fake_run(*a, **k):
        return subprocess.CompletedProcess(a, 0, stdout="19.7.0\n", stderr="")

    monkeypatch.setattr(dep.subprocess, "run", fake_run)
    info = check_portaudio()
    assert info.installed is True
    assert info.version == "19.7.0"


def test_check_portaudio_pkgconfig_fail_falls_through(monkeypatch):
    # pkg-config present but returns nonzero -> falls to sounddevice import
    monkeypatch.setattr(dep.shutil, "which", lambda t: "/usr/bin/pkg-config")

    def fake_run(*a, **k):
        return subprocess.CompletedProcess(a, 1, stdout="", stderr="no")

    monkeypatch.setattr(dep.subprocess, "run", fake_run)
    info = check_portaudio()
    # Result depends on whether sounddevice imports; just assert it's a DependencyInfo
    assert isinstance(info, DependencyInfo)
    assert info.name == "PortAudio"


# ----------------------------------------------------------------------------
# install command tables
# ----------------------------------------------------------------------------
def test_get_install_commands_known():
    cmds = get_install_commands("debian")
    assert cmds["ffmpeg"] == "sudo apt install ffmpeg"


def test_get_install_commands_unknown():
    assert get_install_commands("plan9") == {}


def test_get_full_install_command_known():
    assert "apt install" in get_full_install_command("debian")


def test_get_full_install_command_unknown():
    assert get_full_install_command("plan9") is None


# ----------------------------------------------------------------------------
# check_all
# ----------------------------------------------------------------------------
def test_check_all(monkeypatch):
    monkeypatch.setattr(dep, "detect_os", lambda: "debian")
    monkeypatch.setattr(
        dep,
        "check_ffmpeg",
        lambda: DependencyInfo("FFmpeg", "", "", True),
    )
    monkeypatch.setattr(
        dep,
        "check_libsndfile",
        lambda: DependencyInfo("libsndfile", "", "", True),
    )
    monkeypatch.setattr(
        dep,
        "check_portaudio",
        lambda: DependencyInfo("PortAudio", "", "", False),
    )
    report = check_all()
    assert isinstance(report, DependencyReport)
    assert report.os_type == "debian"
    assert report.all_installed is False
    assert report.install_command is not None
    # install_hint populated from table for ffmpeg
    ffmpeg = [d for d in report.dependencies if d.name == "FFmpeg"][0]
    assert ffmpeg.install_hint == "sudo apt install ffmpeg"


# ----------------------------------------------------------------------------
# run_install / install
# ----------------------------------------------------------------------------
def test_run_install_default_os_detect(monkeypatch):
    # os_type=None -> calls detect_os internally
    monkeypatch.setattr(dep, "detect_os", lambda: "windows")
    ok, msg = run_install()
    assert ok is False
    assert "Windows" in msg


def test_run_install_windows():
    ok, msg = run_install("windows")
    assert ok is False
    assert "Windows" in msg


def test_run_install_unknown():
    ok, msg = run_install("unknown")
    assert ok is False
    assert "Could not detect" in msg


def test_run_install_no_command():
    # os_type with no full install command
    ok, msg = run_install("freebsd")
    assert ok is False


def test_run_install_requires_sudo(monkeypatch):
    monkeypatch.setattr(dep, "_has_sudo", lambda: False)
    ok, msg = run_install("debian")
    assert ok is False
    assert "sudo" in msg


def test_run_install_success(monkeypatch):
    monkeypatch.setattr(dep, "_has_sudo", lambda: True)

    def fake_run(*a, **k):
        return subprocess.CompletedProcess(a, 0, stdout="ok", stderr="")

    monkeypatch.setattr(dep.subprocess, "run", fake_run)
    ok, msg = run_install("debian")
    assert ok is True
    assert "successfully" in msg


def test_run_install_macos_success(monkeypatch):
    def fake_run(*a, **k):
        return subprocess.CompletedProcess(a, 0, stdout="ok", stderr="")

    monkeypatch.setattr(dep.subprocess, "run", fake_run)
    ok, msg = run_install("macos")
    assert ok is True


def test_run_install_failure_returncode(monkeypatch):
    monkeypatch.setattr(dep, "_has_sudo", lambda: True)

    def fake_run(*a, **k):
        return subprocess.CompletedProcess(a, 1, stdout="", stderr="permission denied")

    monkeypatch.setattr(dep.subprocess, "run", fake_run)
    ok, msg = run_install("debian")
    assert ok is False
    assert "permission denied" in msg


def test_run_install_timeout(monkeypatch):
    monkeypatch.setattr(dep, "_has_sudo", lambda: True)

    def fake_run(*a, **k):
        raise subprocess.TimeoutExpired(cmd="apt", timeout=300)

    monkeypatch.setattr(dep.subprocess, "run", fake_run)
    ok, msg = run_install("debian")
    assert ok is False
    assert "timed out" in msg


def test_run_install_filenotfound(monkeypatch):
    monkeypatch.setattr(dep, "_has_sudo", lambda: True)

    def fake_run(*a, **k):
        raise FileNotFoundError("apt missing")

    monkeypatch.setattr(dep.subprocess, "run", fake_run)
    ok, msg = run_install("debian")
    assert ok is False
    assert "not found" in msg


def test_run_install_generic_exception(monkeypatch):
    monkeypatch.setattr(dep, "_has_sudo", lambda: True)

    def fake_run(*a, **k):
        raise RuntimeError("weird")

    monkeypatch.setattr(dep.subprocess, "run", fake_run)
    ok, msg = run_install("debian")
    assert ok is False
    assert "failed" in msg.lower()


def test_install_raises_on_failure(monkeypatch):
    monkeypatch.setattr(dep, "run_install", lambda os_type=None: (False, "nope"))
    with pytest.raises(ProcessingError, match="nope"):
        install()


def test_install_success(monkeypatch):
    monkeypatch.setattr(dep, "run_install", lambda os_type=None: (True, "done"))
    assert install() == "done"


def test_has_sudo_win32(monkeypatch):
    monkeypatch.setattr(dep.sys, "platform", "win32")
    assert dep._has_sudo() is False


def test_has_sudo_unix(monkeypatch):
    monkeypatch.setattr(dep.sys, "platform", "linux")

    def fake_run(*a, **k):
        return subprocess.CompletedProcess(a, 0)

    monkeypatch.setattr(dep.subprocess, "run", fake_run)
    assert dep._has_sudo() is True


def test_has_sudo_exception(monkeypatch):
    monkeypatch.setattr(dep.sys, "platform", "linux")

    def fake_run(*a, **k):
        raise OSError("boom")

    monkeypatch.setattr(dep.subprocess, "run", fake_run)
    assert dep._has_sudo() is False
