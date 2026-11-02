"""The catalog keys are read from a .env file in the working directory or a parent."""

import json
import os
import subprocess
import sys

import pytest

KEYS = ("EBIRD_API_KEY", "XC_API_KEY", "HF_TOKEN", "BIOAMLA_REMOTE", "OTHER_SECRET")


def _run_import(cwd, script, env=None, extra_code=""):
    """Run ``import bioamla`` from ``cwd`` in a script stored at ``script``, like the
    ``bioamla`` command does (under ``python -c`` python-dotenv searches from the working
    directory anyway, hiding the difference). Returns the completed process."""
    script.write_text(
        f"import json, os{extra_code}\nimport bioamla\n"
        f"print(json.dumps({{k: os.environ.get(k) for k in {KEYS!r}}}))\n"
    )
    run_env = {k: v for k, v in os.environ.items() if k not in KEYS}
    run_env.update(env or {})
    return subprocess.run([sys.executable, str(script)], cwd=cwd, env=run_env,
                          capture_output=True, text=True, timeout=60)  # fmt: skip


def _env_after_import(cwd, scripts, env=None):
    out = _run_import(cwd, scripts / "run.py", env)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.splitlines()[-1])


def _project(tmp_path):
    project, scripts = tmp_path / "project", tmp_path / "elsewhere"
    (project / "sub").mkdir(parents=True)
    scripts.mkdir()
    (project / ".env").write_text(
        "XC_API_KEY=from-file\nBIOAMLA_REMOTE=lab:ws\nOTHER_SECRET=nope\n"
    )
    return project, scripts


def test_catalog_keys_come_from_a_dotenv_in_the_working_directory_or_a_parent(tmp_path):
    project, scripts = _project(tmp_path)
    for cwd in (project, project / "sub"):
        seen = _env_after_import(cwd, scripts)
        assert seen["XC_API_KEY"] == "from-file"
        # Only the catalog keys: nothing else in the file is loaded.
        assert seen["BIOAMLA_REMOTE"] is None and seen["OTHER_SECRET"] is None


def test_an_exported_variable_wins_over_the_file(tmp_path):
    project, scripts = _project(tmp_path)
    seen = _env_after_import(project, scripts, {"XC_API_KEY": "exported"})
    assert seen["XC_API_KEY"] == "exported"


@pytest.mark.skipif(sys.platform == "win32", reason="Windows won't remove the current folder")
def test_a_deleted_working_directory_doesnt_break_the_import(tmp_path):
    gone = tmp_path / "gone"
    gone.mkdir()
    out = _run_import(tmp_path, tmp_path / "run.py", {"GONE": str(gone)},
                      extra_code="\nos.chdir(os.environ['GONE']); os.rmdir(os.environ['GONE'])")  # fmt: skip
    assert out.returncode == 0, out.stderr
    assert (
        "Couldn't read" not in out.stderr and "Skipping" not in out.stderr
    )  # nothing to warn about


def _skipped(project, scripts):
    """Run the import against the project's .env; it must be skipped with a warning and
    leave the key unset."""
    out = _run_import(project, scripts / "run.py")
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.splitlines()[-1])["XC_API_KEY"] is None
    return out.stderr


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs a denied read")
def test_an_unreadable_dotenv_is_skipped_with_a_warning(tmp_path):
    project, scripts = _project(tmp_path)
    (project / ".env").chmod(0o000)
    try:
        stderr = _skipped(project, scripts)
    finally:
        (project / ".env").chmod(0o644)
    assert "Couldn't read" in stderr and ".env" in stderr


def test_a_dotenv_that_isnt_utf8_is_skipped_with_a_warning(tmp_path):
    project, scripts = _project(tmp_path)
    (project / ".env").write_bytes(b"XC_API_KEY=\xff\xfe\n")
    assert "Couldn't read" in _skipped(project, scripts)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
def test_a_fifo_named_dotenv_is_skipped_not_opened(tmp_path):
    project, scripts = _project(tmp_path)
    (project / ".env").unlink()
    os.mkfifo(project / ".env")  # opening it would block forever
    try:
        stderr = _skipped(project, scripts)
    except subprocess.TimeoutExpired:
        pytest.fail("import bioamla blocked opening a FIFO named .env")
    assert "not a regular file" in stderr
