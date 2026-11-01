"""Model loading can't hit transformers' threaded-loading deadlock."""

import subprocess
import sys
import threading

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


def test_a_model_loads_many_times_in_a_row(tiny_ast, test_audio_path):
    from bioamla.ml.inference import ASTInference

    done = []

    def load_repeatedly():
        for _ in range(15):
            ASTInference(model_path=str(tiny_ast)).predict(test_audio_path)
            done.append(1)

    worker = threading.Thread(target=load_repeatedly, daemon=True)
    worker.start()
    worker.join(timeout=300)
    assert len(done) == 15, f"loading hung after {len(done)} loads"
