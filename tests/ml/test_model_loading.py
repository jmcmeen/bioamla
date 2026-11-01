"""Model loading can't hit transformers' threaded-loading deadlock."""

import subprocess
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")


def _env_after_import(preset: str | None) -> str:
    code = (
        "import os\n"
        + (f"os.environ['HF_DEACTIVATE_ASYNC_LOAD'] = {preset!r}\n" if preset is not None else
           "os.environ.pop('HF_DEACTIVATE_ASYNC_LOAD', None)\n")
        + "import bioamla.ml\nprint(os.environ['HF_DEACTIVATE_ASYNC_LOAD'])"
    )  # fmt: skip
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_weights_load_serially_unless_asked_otherwise():
    assert _env_after_import(None) == "1"
    assert _env_after_import("0") == "0"  # a user's choice stands


@pytest.mark.slow
def test_a_model_loads_many_times_in_a_row(tiny_ast, test_audio_path):
    """In a fresh process with bioamla's default (a hang there fails the test after the
    timeout instead of hanging the suite; a deadlock holding the GIL would freeze a
    thread in this one)."""
    code = (
        "import os, sys\n"
        "os.environ.pop('HF_DEACTIVATE_ASYNC_LOAD', None)\n"
        "from bioamla.ml.inference import ASTInference\n"
        "for _ in range(15):\n"
        "    ASTInference(model_path=sys.argv[1]).predict(sys.argv[2])\n"
        "print('ok')"
    )
    try:
        out = subprocess.run([sys.executable, "-c", code, str(tiny_ast), test_audio_path],
                             capture_output=True, text=True, timeout=300)  # fmt: skip
    except subprocess.TimeoutExpired:
        pytest.fail("loading the model hung")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")
