"""A model's own feature extractor, and the warning when it has none."""

import logging
import shutil

import pytest

pytest.importorskip("transformers")


def test_a_model_without_an_extractor_gets_the_default_with_a_warning(tiny_ast, tmp_path, caplog):
    from bioamla.ml import get_cached_feature_extractor, load_feature_extractor

    bare = tmp_path / "bare"
    shutil.copytree(tiny_ast, bare)
    (bare / "preprocessor_config.json").unlink()
    with caplog.at_level(logging.WARNING, logger="bioamla.ml.ast"):
        fallback = load_feature_extractor(str(bare))
    assert fallback.max_length == 1024  # the default, not the model's 128
    assert "no feature extractor" in caplog.text

    # The fallback isn't cached: once the model's extractor is saved, it's used.
    assert get_cached_feature_extractor(str(bare)).max_length == 1024
    shutil.copy(tiny_ast / "preprocessor_config.json", bare)
    assert get_cached_feature_extractor(str(bare)).max_length == 128
    assert load_feature_extractor(str(tiny_ast)).max_length == 128
