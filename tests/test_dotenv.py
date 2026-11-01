"""A .env file in the working directory (or a parent) is loaded on import."""

import os
import subprocess
import sys


def _key_after_import(cwd, scripts, env=None):
    """BIOAMLA_TEST_KEY as a script in ``scripts`` sees it after ``import bioamla``,
    run from ``cwd``. A script file, like the ``bioamla`` command: under ``python -c``
    python-dotenv always searches from the working directory, hiding the difference."""
    script = scripts / "run.py"
    script.write_text("import os, bioamla\nprint(os.environ.get('BIOAMLA_TEST_KEY', ''))\n")
    run_env = {k: v for k, v in os.environ.items() if k != "BIOAMLA_TEST_KEY"}
    run_env.update(env or {})
    out = subprocess.run([sys.executable, str(script)], cwd=cwd, env=run_env,
                         capture_output=True, text=True, check=True)  # fmt: skip
    return out.stdout.strip()


def _project(tmp_path):
    project, scripts = tmp_path / "project", tmp_path / "elsewhere"
    (project / "sub").mkdir(parents=True)
    scripts.mkdir()
    (project / ".env").write_text("BIOAMLA_TEST_KEY=from-file\n")
    return project, scripts


def test_a_dotenv_in_the_working_directory_or_a_parent_is_loaded(tmp_path):
    project, scripts = _project(tmp_path)
    assert _key_after_import(project, scripts) == "from-file"
    assert _key_after_import(project / "sub", scripts) == "from-file"


def test_an_exported_variable_wins_over_the_file(tmp_path):
    project, scripts = _project(tmp_path)
    assert _key_after_import(project, scripts, {"BIOAMLA_TEST_KEY": "exported"}) == "exported"
