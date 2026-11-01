"""The crop step's box helpers (no bioamla run needed)."""

import importlib.util

from tests.skills.conftest import skill_script


def load_crops(monkeypatch):
    monkeypatch.syspath_prepend(str(skill_script("label", "crops").parent))
    spec = importlib.util.spec_from_file_location("label_crops", skill_script("label", "crops"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BOX = {"start": 0.25, "end": 0.75, "low_freq": 500.0, "high_freq": 5000.0}


def test_box_option_formats_the_box_for_audio_visualize(monkeypatch):
    crops = load_crops(monkeypatch)
    assert crops.box_option(BOX, "frog") == "0.25,0.75,500.0,5000.0,frog"
    assert crops.box_option(BOX, "Frog, toad") == "0.25,0.75,500.0,5000.0,Frog, toad"
    assert crops.box_option({**BOX, "low_freq": None}) == "0.25,0.75"
    assert crops.box_option({**BOX, "low_freq": None}, "frog") == "0.25,0.75,,,frog"
    assert crops.box_option({**BOX, "end": 12345.678901}).split(",")[1] == "12345.678901"


def test_degenerate_boxes_are_not_drawn(monkeypatch):
    crops = load_crops(monkeypatch)
    assert crops.drawable(BOX)
    assert not crops.drawable({**BOX, "end": 0.25})
    assert not crops.drawable({**BOX, "high_freq": 500.0})
    assert crops.drawable({**BOX, "low_freq": None})
