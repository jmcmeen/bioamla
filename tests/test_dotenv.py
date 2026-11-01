"""The catalog keys are read from a .env file in the working directory or a parent."""

import json
import os
import subprocess
import sys

KEYS = ("EBIRD_API_KEY", "XC_API_KEY", "HF_TOKEN", "BIOAMLA_REMOTE", "OTHER_SECRET")


def _env_after_import(cwd, scripts, env=None):
    """The variables in KEYS as a script in ``scripts`` sees them after ``import
    bioamla``, run from ``cwd``. A script file, like the ``bioamla`` command: under
    ``python -c`` python-dotenv searches from the working directory anyway, hiding the
    difference."""
    script = scripts / "run.py"
    script.write_text(
        f"import json, os, bioamla\nprint(json.dumps({{k: os.environ.get(k) for k in {KEYS!r}}}))\n"
    )
    run_env = {k: v for k, v in os.environ.items() if k not in KEYS}
    run_env.update(env or {})
    out = subprocess.run([sys.executable, str(script)], cwd=cwd, env=run_env,
                         capture_output=True, text=True, check=True, timeout=60)  # fmt: skip
    return json.loads(out.stdout)


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


def test_a_deleted_working_directory_doesnt_break_the_import(tmp_path):
    gone = tmp_path / "gone"
    gone.mkdir()
    script = tmp_path / "run.py"
    script.write_text(
        "import os\nos.chdir(os.environ['GONE'])\nos.rmdir(os.environ['GONE'])\n"
        "import bioamla\nprint('ok')\n"
    )
    out = subprocess.run([sys.executable, str(script)], env={**os.environ, "GONE": str(gone)},
                         capture_output=True, text=True, timeout=60)  # fmt: skip
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
