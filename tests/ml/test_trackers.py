"""The MLflow and TensorBoard tracker adapters, with the backends faked."""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("torch")

from bioamla.ml.trackers import MLflowTracker, TensorBoardTracker, flatten_params  # noqa: E402
from bioamla.protocols import ExperimentTracker  # noqa: E402


def test_adapters_satisfy_the_protocol() -> None:
    assert isinstance(MLflowTracker(), ExperimentTracker)
    assert isinstance(TensorBoardTracker("runs"), ExperimentTracker)


def test_flatten_params() -> None:
    flat = flatten_params({"lr": 1e-4, "aug": {"add_noise": True}, "labels": ["a", "b"], "x": None})
    assert flat == {"lr": "0.0001", "aug.add_noise": "True", "labels": '["a", "b"]', "x": "None"}


def _fake_mlflow(active=None) -> MagicMock:
    fake = MagicMock()
    fake.active_run.return_value = active
    return fake


@pytest.mark.parametrize(
    ("status", "mlflow_status"),
    [("finished", "FINISHED"), ("cancelled", "KILLED"), ("failed", "FAILED")],
)
def test_mlflow_run_ends_with_the_real_status(status, mlflow_status) -> None:
    fake = _fake_mlflow()
    with patch.dict("sys.modules", {"mlflow": fake}):
        tracker = MLflowTracker(tracking_uri="file:/tmp/mlruns", experiment="frogs")
        tracker.start_run("run-1", {"lr": 1e-4, "aug": {"gain": True}})
        tracker.log_metrics(3, {"loss": 0.4})
        tracker.log_artifact("/m/bioamla_run.json")
        tracker.end_run(status)
    fake.set_tracking_uri.assert_called_once_with("file:/tmp/mlruns")
    fake.set_experiment.assert_called_once_with("frogs")
    fake.start_run.assert_called_once_with(run_name="run-1", nested=False)
    fake.log_params.assert_called_once_with({"lr": "0.0001", "aug.gain": "True"})
    fake.log_metrics.assert_called_once_with({"loss": 0.4}, step=3)
    fake.log_artifact.assert_called_once_with("/m/bioamla_run.json")
    fake.end_run.assert_called_once_with(status=mlflow_status)


def test_mlflow_nests_inside_an_active_run() -> None:
    fake = _fake_mlflow(active="sweep-parent")
    with patch.dict("sys.modules", {"mlflow": fake}):
        tracker = MLflowTracker()
        tracker.start_run("run-1", {})
        tracker.end_run("failed")
    # Its own child run, so two runs in one sweep don't collide on params.
    fake.start_run.assert_called_once_with(run_name="run-1", nested=True)
    fake.end_run.assert_called_once_with(status="FAILED")


def test_mlflow_truncates_long_param_values() -> None:
    fake = _fake_mlflow()
    with (
        patch.dict("sys.modules", {"mlflow": fake}),
        patch("bioamla.ml.trackers._mlflow_param_limit", return_value=20),
    ):
        MLflowTracker().start_run(None, {"labels": [f"species-{i}" for i in range(50)], "lr": 1})
    sent = fake.log_params.call_args.args[0]
    assert len(sent["labels"]) == 20 and sent["labels"].endswith("...")
    assert sent["lr"] == "1"


def test_tensorboard_writes_scalars_text_and_closes(tmp_path) -> None:
    writer = MagicMock()
    with patch("torch.utils.tensorboard.SummaryWriter", return_value=writer) as cls:
        tracker = TensorBoardTracker(str(tmp_path))
        tracker.start_run("run-1", {"lr": 1e-4})
        tracker.log_metrics(5, {"loss": 0.2, "eval_accuracy": 0.9})
        tracker.log_artifact("/m/bioamla_run.json")
        tracker.end_run("cancelled")
    cls.assert_called_once_with(log_dir=str(tmp_path))
    writer.add_scalar.assert_any_call("loss", 0.2, 5)
    writer.add_scalar.assert_any_call("eval_accuracy", 0.9, 5)
    writer.add_text.assert_any_call("status", "cancelled", 5)
    writer.close.assert_called_once()
