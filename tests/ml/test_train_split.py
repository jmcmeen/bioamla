"""Tests for fixed-split consumption in AST training (partition --mode column)."""

import pandas as pd

from bioamla.ml.training import _dataframe_to_ast_dataset
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


class TestSplitOffEval:
    """The 20% eval split training makes when a dataset has no split of its own."""

    @staticmethod
    def _labeled(counts):
        labels = [label for label, n in enumerate(counts) for _ in range(n)]
        return Dataset.from_dict({"audio": [f"/x/{i}.wav" for i in range(len(labels))],
                                  "labels": labels})  # fmt: skip

    def test_split_is_stratified(self, caplog):
        from collections import Counter

        from bioamla.ml.training import _split_off_eval

        # 90 / 10 imbalance: a plain shuffled split (seed 0) gives the rare class 3 of
        # the 20 eval rows; stratified, each class gives exactly 20% of its rows.
        with caplog.at_level("WARNING", logger="bioamla.ml.training"):
            split = _split_off_eval(self._labeled([90, 10]), 2)

        assert Counter(split["test"]["labels"]) == {0: 18, 1: 2}
        assert Counter(split["train"]["labels"]) == {0: 72, 1: 8}
        assert "Stratified split failed" not in caplog.text

    def test_rows_are_selected_not_rewritten(self):
        from bioamla.ml.training import _split_off_eval

        # The other columns come through untouched, each row once, and ``labels``
        # keeps its plain integer type (no cast of the dataset itself).
        ds = self._labeled([10, 10])
        split = _split_off_eval(ds, 2)

        assert split["train"].features == ds.features
        rows = sorted([*split["train"]["audio"], *split["test"]["audio"]])
        assert rows == sorted(ds["audio"])
        by_audio = dict(zip(list(ds["audio"]), list(ds["labels"]), strict=True))
        assert all(by_audio[r["audio"]] == r["labels"] for r in split["test"])

    def test_labels_stay_integer_ids(self):
        from bioamla.ml.training import _split_off_eval

        split = _split_off_eval(self._labeled([10, 10, 10]), 3)
        assert set(split["train"]["labels"]) == {0, 1, 2}
        assert all(isinstance(v, int) for v in split["train"]["labels"])

    def test_single_example_class_falls_back_with_a_warning(self, caplog):
        from bioamla.ml.training import _split_off_eval

        with caplog.at_level("WARNING", logger="bioamla.ml.training"):
            split = _split_off_eval(self._labeled([9, 1]), 2)

        assert len(split["train"]) + len(split["test"]) == 10
        assert "Stratified split failed" in caplog.text
