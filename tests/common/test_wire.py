"""The wire form: results as JSON that turns back into the same objects."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from bioamla.common.wire import WireError, from_wire, to_wire
from bioamla.workspace import Artifact


def round_trip(value):
    return from_wire(json.loads(json.dumps(to_wire(value), allow_nan=False)))


def test_plain_values_round_trip():
    value = {"a": [1, 2.5, "x", None, True], "t": (1, (2, 3)), "p": Path("a/b.wav"),
             "nan": float("nan"), "inf": float("-inf"), 3: "int key", "__wire__": "tag-like"}  # fmt: skip
    back = round_trip(value)
    assert back["t"] == (1, (2, 3)) and back["p"] == Path("a/b.wav") and back[3] == "int key"
    assert math.isnan(back["nan"]) and back["inf"] == float("-inf")
    assert back["__wire__"] == "tag-like" and back["a"] == value["a"]


def test_numpy_round_trips_with_dtype_and_shape():
    arr = np.arange(6, dtype="int16").reshape(2, 3)
    back = round_trip({"arr": arr, "s": np.float32(1.5), "b": np.bool_(True)})
    assert back["arr"].dtype == np.int16 and back["arr"].shape == (2, 3)
    assert (back["arr"] == arr).all()
    assert back["s"].dtype == np.float32 and back["s"] == 1.5
    assert isinstance(back["b"], np.bool_) and bool(back["b"])


def test_bioamla_dataclasses_round_trip_without_rerunning_init():
    artifact = Artifact(id="a1", kind="audio_file", mode="linked", path="rec.wav")
    back = round_trip([artifact])
    assert back == [artifact] and type(back[0]) is Artifact


def test_real_results_round_trip(test_audio_path):
    from bioamla import operations

    for name, request in (("audio.info", {"file": test_audio_path}),
                          ("detect.energy", {"file": test_audio_path})):  # fmt: skip
        try:
            result = operations.run(name, request)
        except Exception as e:  # noqa: BLE001 - an extra may be missing
            pytest.skip(str(e))
        back = round_trip(result)
        assert type(back) is type(result)
        assert back.to_dict() == result.to_dict()


@dataclass
class NotOurs:
    x: int = 1
    y: list = field(default_factory=list)


def test_only_bioamla_types_go_out_or_come_back():
    with pytest.raises(WireError):
        to_wire(NotOurs())
    with pytest.raises(WireError):
        to_wire(object())
    evil = ("os:system", "subprocess:Popen", "builtins:eval", "bioamla:nothing_here",
            # Classes a bioamla module imports are not bioamla's:
            "bioamla.system.dependency:subprocess.Popen", "bioamla.common.wire:Path")  # fmt: skip
    for name in evil:
        for kind in ("dataclass", "enum"):
            with pytest.raises(WireError):
                from_wire({"__wire__": kind, "type": name, "fields": {}, "value": ["sh"]})
    with pytest.raises(WireError):  # a bioamla dataclass is not an enum
        from_wire({"__wire__": "enum", "type": "bioamla.workspace:Artifact", "value": "x"})
    with pytest.raises(WireError):  # a bioamla function, not a dataclass
        from_wire({"__wire__": "dataclass", "type": "bioamla.workspace:open_workspace",
                   "fields": {}})  # fmt: skip
    with pytest.raises(WireError):
        from_wire({"__wire__": "nope"})


def test_arrays_travel_compactly():
    arr = np.zeros((100, 100), dtype="float32")
    wire = to_wire(arr)
    assert "b64" in wire and len(json.dumps(wire)) < 60_000
    with pytest.raises(WireError):
        to_wire(np.array([object()], dtype=object))


def test_dataframes_round_trip_with_dtypes_and_index():
    import pandas as pd

    frame = pd.DataFrame(
        {"t": np.array([0.0, 0.5], dtype="float32"), "n": [1, 2], "s": ["a", "b"],
         "x": [float("nan"), 1.0]},
        index=[10, 20],
    )  # fmt: skip
    back = round_trip({"table": frame})["table"]
    pd.testing.assert_frame_equal(back, frame)
    empty = pd.DataFrame(columns=["a"]).astype({"a": "float64"})
    pd.testing.assert_frame_equal(round_trip(empty), empty, check_index_type=False)


def test_a_result_holding_a_table_round_trips(test_audio_path):
    import pandas as pd

    from bioamla import operations

    try:
        result = operations.run("audio.features", {"file": test_audio_path})
    except Exception as e:  # noqa: BLE001 - an extra may be missing
        pytest.skip(str(e))
    back = round_trip(result)
    pd.testing.assert_frame_equal(back.table, result.table)


def test_dataframes_keep_missing_values_timedeltas_and_their_index():
    import pandas as pd

    frame = pd.DataFrame(
        {"n": pd.array([1, None], dtype="Int64"), "s": pd.array(["a", None], dtype="string"),
         "when": pd.to_datetime(["2024-01-01", None]),
         "gap": pd.to_timedelta([1.5, None], unit="s")},
        index=pd.Index(["x", "y"], name="file"),
    )  # fmt: skip
    pd.testing.assert_frame_equal(round_trip(frame), frame)
    rows_only = pd.DataFrame(index=pd.Index(["a", "b"], name="f"))
    back = round_trip(rows_only)
    assert back.shape == (2, 0) and list(back.index) == ["a", "b"] and back.index.name == "f"
