"""Eval for the train skill: check a config, train, evaluate, keep or discard."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tests.skills.conftest import SKILLS, SR


def load(name: str):
    """A train skill script as a module (its folder on the path, as when run)."""
    folder = SKILLS / "train" / "scripts"
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location(f"train_{name}", folder / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(folder))


def evaluation(f1: float, per_class: dict[str, float], **eval_set) -> dict:
    return {
        "evaluated": True,
        "model": "m",
        "eval_set": {
            "audio": "eval/audio",
            "ground_truth": "eval/labels.csv",
            "ground_truth_sha256": "abc",
            "file_column": "file_name",
            "label_column": "label",
            "total_samples": 20,
            "skipped": [],
            **eval_set,
        },  # fmt: skip
        "metrics": {"accuracy": f1, "precision": f1, "recall": f1, "f1_score": f1},
        "per_class": {
            label: {"precision": v, "recall": v, "f1_score": v, "support": 10}
            for label, v in per_class.items()
        },
    }


def put(root: Path, run: str, name: str, data: dict) -> str:
    folder = root / run
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(data))
    return run


# --- compare.py: the verdict, from JSON alone -------------------------------------


def test_compare_keeps_a_better_run(run_skill, workspace) -> None:
    base = put(workspace, "runs/train/base", "eval.json", evaluation(0.70, {"a": 0.7, "b": 0.7}))
    cand = put(workspace, "runs/train/cand", "eval.json", evaluation(0.80, {"a": 0.8, "b": 0.8}))
    code, out = run_skill("train", "compare", cand, "--baseline", base, cwd=workspace)
    assert code == 0, out
    assert out["keep"] and out["comparable"] and out["reasons"] == []
    assert out["delta"] == pytest.approx(0.10)
    assert out["class_f1_changes"] == {"a": pytest.approx(0.1), "b": pytest.approx(0.1)}
    saved = json.loads((workspace / cand / "compare.json").read_text())
    assert saved["keep"] is True and saved["baseline"] == base


def test_compare_discards_by_each_threshold(run_skill, workspace) -> None:
    base = put(workspace, "runs/train/base", "eval.json", evaluation(0.70, {"a": 0.9, "b": 0.5}))
    # Better overall, but class a fell by 0.2.
    cand = put(workspace, "runs/train/cand", "eval.json", evaluation(0.75, {"a": 0.7, "b": 0.8}))
    code, out = run_skill("train", "compare", cand, "--baseline", base, cwd=workspace)
    assert code == 2 and not out["keep"]
    assert out["reasons"] == ["a: F1 fell by 0.2000 (at most 0.0500 allowed)"]
    code, out = run_skill("train", "compare", cand, "--baseline", base, "--max-class-drop", "0.25",
                          cwd=workspace)  # fmt: skip
    assert code == 0, out
    # Not better by enough.
    code, out = run_skill("train", "compare", cand, "--baseline", base, "--max-class-drop", "1",
                          "--min-delta", "0.1", cwd=workspace)  # fmt: skip
    assert code == 2 and "f1_score 0.7500 vs the baseline's 0.7000" in out["reasons"][0]
    # A class the candidate never predicts has F1 0.
    gone = put(workspace, "runs/train/gone", "eval.json", evaluation(0.9, {"b": 0.9}))
    code, out = run_skill("train", "compare", gone, "--baseline", base, cwd=workspace)
    assert code == 2 and out["class_f1_changes"]["a"] == pytest.approx(-0.9)


def test_compare_refuses_runs_evaluated_on_different_sets(run_skill, workspace) -> None:
    base = put(workspace, "runs/train/base", "eval.json", evaluation(0.7, {"a": 0.7}))
    for change in ({"ground_truth_sha256": "def"}, {"skipped": ["x.wav"]}, {"audio": "other"},
                   {"audio_sha256": "456"}, {"total_samples": 19}):  # fmt: skip
        cand = put(workspace, "runs/train/cand", "eval.json", evaluation(0.9, {"a": 0.9}, **change))
        code, out = run_skill("train", "compare", cand, "--baseline", base, cwd=workspace)
        assert code == 2 and out["comparable"] is False and not out["keep"]
        assert next(iter(change)) in out["reasons"][0]


def test_compare_keeps_earlier_verdicts(run_skill, workspace) -> None:
    base = put(workspace, "runs/train/base", "eval.json", evaluation(0.7, {"a": 0.9}))
    cand = put(workspace, "runs/train/cand", "eval.json", evaluation(0.8, {"a": 0.7}))
    code, first = run_skill("train", "compare", cand, "--baseline", base, cwd=workspace)
    assert code == 2
    code, second = run_skill("train", "compare", cand, "--baseline", base, "--max-class-drop",
                             "1", cwd=workspace)  # fmt: skip
    assert code == 0 and second["keep"]
    # The looser second verdict shows the discard before it.
    assert [v["keep"] for v in second["earlier"]] == [False]
    assert second["earlier"][0]["thresholds"]["max_class_drop"] == 0.05
    code, third = run_skill("train", "compare", cand, "--baseline", base, cwd=workspace)
    assert [v["keep"] for v in third["earlier"]] == [False, True]


def test_compare_needs_both_evaluations_and_sane_thresholds(run_skill, workspace) -> None:
    base = put(workspace, "runs/train/base", "eval.json", evaluation(0.7, {"a": 0.7}))
    code, out = run_skill("train", "compare", "runs/train/none", "--baseline", base,
                          cwd=workspace)  # fmt: skip
    assert code == 1 and "evaluate.py" in out["error"]
    code, out = run_skill("train", "compare", base, "--baseline", base, "--min-delta", "2",
                          cwd=workspace)  # fmt: skip
    assert code == 1 and "from 0 to 1" in out["error"]


# --- the metrics summary ------------------------------------------------------------


def test_summarise_reads_the_metrics_not_the_logs(tmp_path) -> None:
    train = load("_train")
    lines = [
        {"event": "start", "name": "r", "params": {"run_id": "abc"}},
        {"event": "metrics", "step": 5, "metrics": {"loss": 0.9}},
        {"event": "metrics", "step": 10, "metrics": {"eval_loss": 0.5, "eval_accuracy": 0.8,
                                                     "eval_runtime": 3.0}},
        {"event": "metrics", "step": 15, "metrics": {"loss": 0.4}},
        {"event": "metrics", "step": 20, "metrics": {"eval_loss": 0.6, "eval_accuracy": 0.7,
                                                     "eval_runtime": 9.0}},
        {"event": "end", "status": "finished"},
    ]  # fmt: skip
    path = tmp_path / "m.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + '\n{"event": "metr')  # cut short
    summary = train.summarise(train.read_metrics(path))
    assert summary == {
        "status": "finished", "run_id": "abc", "steps": 20, "train_loss": 0.4,
        "final_eval": {"eval_loss": 0.6, "eval_accuracy": 0.7},
        "best_eval": {"eval_loss": 0.5, "eval_accuracy": 0.8},  # lowest loss, highest accuracy
    }  # fmt: skip
    assert train.summarise([{"event": "metrics", "step": 1, "metrics": {}}])["status"] is None


# --- config.py and train.py refusals (no training) -----------------------------------


def write_config(workspace: Path, text: str) -> str:
    (workspace / "train.toml").write_text(text)
    return "train.toml"


def class_folder(workspace: Path) -> str:
    t = np.arange(SR) / SR
    for label, freq in (("a", 1000), ("b", 3000)):
        for i in range(4):
            path = workspace / "data" / label / f"{i}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, (0.5 * np.sin(2 * np.pi * freq * (i + 1) * t / 2)).astype(np.float32),
                     SR)  # fmt: skip
    return "data"


def test_config_refuses_an_invalid_config(run_skill, workspace) -> None:
    config = write_config(workspace, "[training]\nepochs = 0\n")
    code, out = run_skill("train", "config", config, "--dataset", "org/data", cwd=workspace)
    assert code == 2 and not out["valid"]
    assert "epochs" in out["reason"]
    saved = json.loads((workspace / out["run_dir"] / "config.json").read_text())
    assert saved["valid"] is False
    # An inverted augmentation range, and a training CSV that isn't there.
    config = write_config(workspace, "[augmentation]\nadd_noise = true\nmin_snr_db = 30.0\n"
                                     "max_snr_db = 10.0\n")  # fmt: skip
    code, out = run_skill("train", "config", config, "--dataset", "org/data", cwd=workspace)
    assert code == 2 and "SNR" in out["reason"]
    config = write_config(workspace, "[training]\nepochs = 1\n")
    code, out = run_skill("train", "config", config, "--dataset", "typo.csv", cwd=workspace)
    assert code == 2 and "typo.csv" in out["reason"]
    # The refused run can't train.
    code, out = run_skill("train", "train", out["run_dir"], cwd=workspace)
    assert code == 2 and "wasn't valid" in out["reason"]


def test_config_records_the_options_and_train_refuses_a_changed_config(
    run_skill, workspace
) -> None:
    config = write_config(workspace, "[training]\nepochs = 3\n[augmentation]\nmultiplier = 2\n")
    code, out = run_skill("train", "config", config, "--dataset", class_folder(workspace),
                          cwd=workspace)  # fmt: skip
    assert code == 0, out
    assert out["valid"] and out["options"]["num_train_epochs"] == 3
    assert "augment_multiplier 2 ignored" in out["warnings"][0]
    assert out["metrics_file"] == f"{out['run_dir']}/metrics.jsonl"
    assert out["model"] == f"{out['run_dir']}/model/best_model"
    assert not (workspace / out["run_dir"] / "model").exists()  # nothing trained
    write_config(workspace, "[training]\nepochs = 30\n")
    code, refused = run_skill("train", "train", out["run_dir"], cwd=workspace)
    assert code == 2 and "changed since the check" in refused["reason"]
    assert not (workspace / out["run_dir"] / "train.json").exists()


def test_a_failed_step_leaves_no_run_folder(run_skill, workspace) -> None:
    code, out = run_skill("train", "config", "missing.toml", "--dataset", "org/data",
                          cwd=workspace)  # fmt: skip
    assert code == 1 and "missing.toml" in out["error"]
    code, out = run_skill("train", "evaluate", "--model", "m", "--audio", "nowhere",
                          "--ground-truth", "missing.csv", cwd=workspace)  # fmt: skip
    assert code == 1
    assert not list((workspace / "runs" / "train").iterdir())


def test_train_collects_a_run_that_started_rather_than_training_again(run_skill, workspace) -> None:
    config = write_config(workspace, "[training]\nepochs = 1\n")
    code, checked = run_skill("train", "config", config, "--dataset", class_folder(workspace),
                              cwd=workspace)  # fmt: skip
    assert code == 0, checked
    run = workspace / checked["run_dir"]
    # As if an earlier train.py stopped waiting while training went on.
    metrics = workspace / checked["metrics_file"]
    metrics.write_text(json.dumps({"event": "start", "name": "r", "params": {"run_id": "x"}})
                       + "\n")  # fmt: skip
    code, out = run_skill("train", "train", checked["run_dir"], cwd=workspace)
    assert code == 2 and out["status"] == "running" and "again later" in out["reason"]
    assert not (run / "train.json").exists()
    with metrics.open("a") as f:
        f.write(json.dumps({"event": "metrics", "step": 4, "metrics": {"eval_accuracy": 0.5}})
                + "\n" + json.dumps({"event": "end", "status": "finished"}) + "\n")  # fmt: skip
    code, out = run_skill("train", "train", checked["run_dir"], cwd=workspace)
    assert code == 0, out
    assert out["trained"] and out["run_id"] == "x" and out["model"] == checked["model"]
    assert out["final_eval"] == {"eval_accuracy": 0.5}
    assert json.loads((run / "train.json").read_text())["trained"] is True


def test_evaluate_refuses_a_run_that_hasnt_trained(run_skill, workspace) -> None:
    put(workspace, "runs/train/r", "config.json", {"valid": True})
    args = ["--audio", "eval", "--ground-truth", "eval.csv"]
    code, out = run_skill("train", "evaluate", "--run", "runs/train/r", *args, cwd=workspace)
    assert code == 2 and "hasn't trained" in out["reason"]
    put(workspace, "runs/train/r", "train.json", {"trained": False, "status": "failed"})
    code, out = run_skill("train", "evaluate", "--run", "runs/train/r", *args, cwd=workspace)
    assert code == 2 and "failed" in out["reason"]
    code, out = run_skill("train", "evaluate", *args, cwd=workspace)
    assert code == 1 and "--run" in out["error"]


# --- the whole loop, with a tiny model ------------------------------------------------


@pytest.mark.integration
def test_train_evaluate_and_compare(run_skill, workspace, tiny_ast) -> None:
    data = class_folder(workspace)
    t = np.arange(SR) / SR
    (workspace / "eval").mkdir()
    for name, freq in (("a.wav", 1000), ("b.wav", 3000)):
        sf.write(workspace / "eval" / name, (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32),
                 SR)  # fmt: skip
    (workspace / "eval.csv").write_text("file_name,label\na.wav,a\nb.wav,b\n")
    config = write_config(
        workspace,
        f"[models]\ndefault_ast_model = '{tiny_ast}'\n"
        "[training]\nepochs = 1\nbatch_size = 4\nreport_to = 'none'\nlogging_steps = 1\n",
    )
    code, checked = run_skill("train", "config", config, "--dataset", data, cwd=workspace)
    assert code == 0, checked
    run = checked["run_dir"]

    code, trained = run_skill("train", "train", run, cwd=workspace)
    assert code == 0, trained
    assert trained["status"] == "finished" and trained["trained"]
    assert trained["model"] == checked["model"] and (workspace / trained["model"]).is_dir()
    assert trained["steps"] >= 1 and trained["train_loss"] is not None
    code, again = run_skill("train", "train", run, cwd=workspace)
    assert code == 2 and "already trained" in again["reason"]

    args = ["--audio", "eval", "--ground-truth", "eval.csv"]
    code, evaluated = run_skill("train", "evaluate", "--run", run, *args, cwd=workspace)
    assert code == 0, evaluated
    assert evaluated["eval_set"]["total_samples"] == 2
    assert set(evaluated["metrics"]) == {"accuracy", "precision", "recall", "f1_score"}
    code, base = run_skill("train", "evaluate", "--model", str(tiny_ast), *args, cwd=workspace)
    assert code == 0, base
    assert base["eval_set"] == evaluated["eval_set"]

    # Against itself: no better, so discarded by default and kept with no margin.
    code, out = run_skill("train", "compare", run, "--baseline", run, cwd=workspace)
    assert code == 2 and out["comparable"] and out["delta"] == 0
    code, out = run_skill("train", "compare", run, "--baseline", run, "--min-delta", "0",
                          cwd=workspace)  # fmt: skip
    assert code == 0 and out["keep"]
    code, out = run_skill("train", "compare", run, "--baseline", base["run_dir"],
                          "--min-delta", "0", "--max-class-drop", "1", cwd=workspace)  # fmt: skip
    assert out["comparable"] and code == (0 if out["delta"] >= 0 else 2)
