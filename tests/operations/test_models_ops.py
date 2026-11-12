"""models.ast.* operations: model names, training options, path checks (no model needed)."""

from pathlib import Path

import pytest

pytest.importorskip("torch")

from bioamla import operations  # noqa: E402
from bioamla.exceptions import InvalidInputError  # noqa: E402
from bioamla.ml.operations import _model  # noqa: E402


def _refusing(allowed):
    """Like the server's check: relative paths are under ``allowed``, and nothing may
    leave it."""

    def check(path):
        full = Path(path) if Path(path).is_absolute() else Path(allowed) / path
        if not str(full).startswith(str(allowed)):
            raise PermissionError(path)
        return str(full)

    return operations.OperationContext(path_check=check)


def test_model_names(tmp_path):
    local = tmp_path / "models" / "frog"
    local.mkdir(parents=True)
    ctx = _refusing(tmp_path)
    assert _model(str(local), ctx) == str(local)  # allowed local path
    assert _model("bioamla/scp-frogs", ctx) == "bioamla/scp-frogs"  # an ID, not a path
    with pytest.raises(PermissionError):
        _model("/etc/models", ctx)  # a path outside the allowed folders


def test_a_refused_id_shaped_name_is_refused(tmp_path):
    """``org/name`` failing the check means it's a link out of the allowed folders: it
    mustn't be retried as a Hugging Face ID, which would load the link's target."""

    def check(path):
        raise PermissionError(path)

    with pytest.raises(PermissionError):
        _model("org/name", operations.OperationContext(path_check=check))


def test_train_checks_paths_from_the_config_file_and_mlflow(tmp_path, mocker):
    train = mocker.patch("bioamla.ml.training.train_ast")
    ctx = _refusing(tmp_path)
    base = {"train_dataset": "org/data", "training_dir": str(tmp_path / "run")}
    config = tmp_path / "train.toml"
    config.write_text('[models]\ndefault_ast_model = "/etc/model"\n')
    with pytest.raises(PermissionError):
        operations.run("models.ast.train", {**base, "config_path": str(config)}, ctx)
    for uri in ("file:///etc/mlruns", "/etc/mlruns"):
        with pytest.raises(PermissionError):
            operations.run(
                "models.ast.train", {**base, "options": {"mlflow_tracking_uri": uri}}, ctx
            )
    train.assert_not_called()
    operations.run("models.ast.train", {**base, "options": {"mlflow_tracking_uri": "mlruns"}}, ctx)
    assert train.call_args.kwargs["mlflow_tracking_uri"] == str(tmp_path / "mlruns")
    remote = "http://tracking:5000"
    operations.run("models.ast.train", {**base, "options": {"mlflow_tracking_uri": remote}}, ctx)
    assert train.call_args.kwargs["mlflow_tracking_uri"] == remote


def test_train_checks_base_model_and_option_types(tmp_path, mocker):
    train = mocker.patch("bioamla.ml.training.train_ast")
    ctx = _refusing(tmp_path)
    base = {"train_dataset": "org/data", "training_dir": str(tmp_path / "run")}
    with pytest.raises(PermissionError):
        operations.run("models.ast.train", {**base, "options": {"base_model": "/etc/x"}}, ctx)
    with pytest.raises(InvalidInputError, match="num_train_epochs"):
        operations.run("models.ast.train", {**base, "options": {"num_train_epochs": "abc"}}, ctx)
    with pytest.raises(InvalidInputError, match="between 0 and 1"):
        operations.run("models.ast.train", {**base, "options": {"noise_probability": 2}}, ctx)
    train.assert_not_called()
    operations.run("models.ast.train", {**base, "options": {"learning_rate": 1}}, ctx)
    assert train.call_args.kwargs["learning_rate"] == 1  # an int is fine for a float


def test_kinds():
    assert operations.get("models.ast.train").kind == "job"
    assert operations.get("models.ast.init-config").kind == "quick"


def test_train_records_to_a_metrics_file(tmp_path, mocker):
    from bioamla.ml.trackers import JsonlTracker

    train = mocker.patch("bioamla.ml.training.train_ast")
    ctx = _refusing(tmp_path)
    base = {"train_dataset": "org/data", "training_dir": str(tmp_path / "run")}
    with pytest.raises(PermissionError):  # a refused request leaves no file
        operations.run(
            "models.ast.train",
            {**base, "metrics_file": "m.jsonl", "options": {"base_model": "/etc/x"}},
            ctx,
        )
    assert not (tmp_path / "m.jsonl").exists()
    operations.run("models.ast.train", {**base, "metrics_file": "run/metrics.jsonl"}, ctx)
    tracker = train.call_args.kwargs["tracker"]
    assert isinstance(tracker, JsonlTracker)
    assert tracker.path == tmp_path / "run" / "metrics.jsonl"  # through the path check
    operations.run("models.ast.train", base, ctx)
    assert train.call_args.kwargs["tracker"] is None
    with pytest.raises(PermissionError):
        operations.run("models.ast.train", {**base, "metrics_file": "/etc/m.jsonl"}, ctx)
    (tmp_path / "old.jsonl").write_text("")
    train.reset_mock()
    with pytest.raises(InvalidInputError, match="already exists"):
        operations.run("models.ast.train", {**base, "metrics_file": "old.jsonl"}, ctx)
    train.assert_not_called()


def test_train_failing_before_training_ends_the_metrics_file(tmp_path, mocker):
    import json

    from bioamla.exceptions import TrainingError

    mocker.patch("bioamla.ml.training.train_ast", side_effect=TrainingError("no rows"))
    ctx = _refusing(tmp_path)
    request = {
        "train_dataset": "org/data",
        "training_dir": str(tmp_path),
        "metrics_file": "m.jsonl",
    }
    with pytest.raises(TrainingError):
        operations.run("models.ast.train", request, ctx)
    lines = (tmp_path / "m.jsonl").read_text().splitlines()
    assert [json.loads(x) for x in lines] == [{"event": "end", "status": "failed"}]


def test_check_train_resolves_and_checks_without_training(tmp_path, mocker):
    from bioamla.exceptions import ConfigValidationError

    train = mocker.patch("bioamla.ml.training.train_ast")
    ctx = _refusing(tmp_path)
    config = tmp_path / "train.toml"
    config.write_text("[training]\nepochs = 7\nlearning_rate = 0.001\n")
    request = {
        "train_dataset": "org/data",
        "training_dir": "run",
        "config_path": str(config),
        "options": {"learning_rate": 0.01},  # a flag wins over the file
        "metrics_file": "run/metrics.jsonl",
    }
    plan = operations.run("models.ast.check-train", request, ctx)
    assert plan.to_dict()["dry_run"] is True
    assert plan.train_dataset == "org/data"
    assert plan.training_dir == str(tmp_path / "run")
    assert plan.metrics_file == str(tmp_path / "run" / "metrics.jsonl")
    assert plan.options["num_train_epochs"] == 7
    assert plan.options["learning_rate"] == 0.01
    assert "training_dir" not in plan.options
    assert plan.warnings == []
    train.assert_not_called()
    assert list(tmp_path.iterdir()) == [config]  # nothing written

    config.write_text("[training]\nepochs = 'many'\n")
    with pytest.raises(ConfigValidationError):
        operations.run("models.ast.check-train", request, ctx)
    config.write_text("[training]\nepochs = 2\n")
    with pytest.raises(PermissionError):  # the same path checks as a run
        operations.run("models.ast.check-train", {**request, "train_dataset": "/etc/data"}, ctx)
    (tmp_path / "run").write_text("")  # the metrics file's folder is a file
    with pytest.raises(InvalidInputError, match="Can't create"):
        operations.run("models.ast.check-train", request, ctx)
    train.assert_not_called()
    assert operations.get("models.ast.check-train").kind == "quick"


def test_check_train_options_makes_the_runs_own_checks(tmp_path):
    from bioamla.exceptions import TrainingError
    from bioamla.ml import check_train_options

    # An inverted range for a layer that's on fails as the run would...
    with pytest.raises(TrainingError, match="SNR"):
        check_train_options("org/data", {"add_noise": True, "min_snr_db": 30.0, "max_snr_db": 10.0})
    # ...but not for one that's off.
    check_train_options("org/data", {"min_snr_db": 30.0, "max_snr_db": 10.0})
    plan = check_train_options("org/data", {"augment_multiplier": 3})
    assert "augment_multiplier 3 ignored" in plan.warnings[0]
    with pytest.raises(TrainingError, match="does not exist"):
        check_train_options(str(tmp_path / "typo.csv"))
    (tmp_path / "notes.txt").write_text("")
    with pytest.raises(TrainingError, match="neither a CSV"):
        check_train_options(str(tmp_path / "notes.txt"))
    (tmp_path / "m.jsonl").write_text("")
    with pytest.raises(InvalidInputError, match="already exists"):
        check_train_options("org/data", metrics_file=tmp_path / "m.jsonl")
    plan = check_train_options(str(tmp_path), metrics_file=tmp_path / "new" / "m.jsonl")
    assert plan.train_dataset == str(tmp_path) and not (tmp_path / "new").exists()
