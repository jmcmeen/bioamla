"""Tests for fixed-split consumption in AST training (partition --mode column)."""

import pandas as pd

from bioamla.cli.commands.models import _dataframe_to_ast_dataset
from datasets import Dataset, DatasetDict


def _df(splits):
    n = len(splits)
    return pd.DataFrame(
        {
            "audio": [f"/x/{i}.wav" for i in range(n)],
            "label": [("call" if i % 2 else "chorus") for i in range(n)],
            "split": splits,
        }
    )


class TestDataframeToAstDataset:
    def test_full_split_builds_datasetdict(self):
        df = _df(["train", "train", "train", "validation", "test", "test"])
        ds, used = _dataframe_to_ast_dataset(df, "label")
        assert used is True
        assert isinstance(ds, DatasetDict)
        assert set(ds.keys()) == {"train", "validation", "test"}
        assert len(ds["train"]) == 3
        assert len(ds["test"]) == 2

    def test_split_aliases_normalized(self):
        # 'val' -> validation, 'eval' -> test, 'training' -> train.
        df = _df(["training", "training", "val", "eval"])
        ds, used = _dataframe_to_ast_dataset(df, "label")
        assert used is True
        assert set(ds.keys()) == {"train", "validation", "test"}

    def test_validation_only_aliased_to_test(self):
        # No explicit test split: validation becomes the eval set.
        df = _df(["train", "train", "val", "val"])
        ds, used = _dataframe_to_ast_dataset(df, "label")
        assert used is True
        assert set(ds.keys()) == {"train", "test"}

    def test_no_split_column_is_flat_dataset(self):
        df = _df(["train"] * 4).drop(columns=["split"])
        ds, used = _dataframe_to_ast_dataset(df, "label")
        assert used is False
        assert isinstance(ds, Dataset)
        assert len(ds) == 4

    def test_empty_split_values_fall_back_to_flat(self):
        df = _df(["", "", "", ""])
        ds, used = _dataframe_to_ast_dataset(df, "label")
        assert used is False
        assert isinstance(ds, Dataset)

    def test_unrecognized_split_falls_back_to_flat(self):
        df = _df(["train", "train", "holdout", "test"])
        ds, used = _dataframe_to_ast_dataset(df, "label")
        assert used is False
        assert isinstance(ds, Dataset)

    def test_no_rows_dropped_in_datasetdict(self):
        df = _df(["train", "train", "validation", "test", "test", "train"])
        ds, _ = _dataframe_to_ast_dataset(df, "label")
        total = sum(len(ds[k]) for k in ds)
        assert total == len(df)
