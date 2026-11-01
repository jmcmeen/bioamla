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
