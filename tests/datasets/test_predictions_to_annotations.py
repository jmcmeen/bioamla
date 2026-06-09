"""Tests for predictions_to_annotations (inference -> reviewable annotations)."""

from bioamla.datasets import predictions_to_annotations


class TestPredictionsToAnnotations:
    def test_segmented_inference_rows(self) -> None:
        # Shape produced by segmented_wave_file_inference.
        rows = [
            {"filepath": "a.wav", "start": 0.0, "stop": 3.0, "prediction": "frog"},
            {"filepath": "a.wav", "start": 3.0, "stop": 6.0, "prediction": "bird"},
        ]
        anns = predictions_to_annotations(rows)
        assert len(anns) == 2
        assert anns[0].start_time == 0.0
        assert anns[0].end_time == 3.0
        assert anns[0].label == "frog"
        assert anns[0].custom_fields["source_file"] == "a.wav"

    def test_alternate_keys_and_confidence(self) -> None:
        rows = [{"start_time": 1.0, "end_time": 2.0, "label": "owl", "confidence": 0.9}]
        anns = predictions_to_annotations(rows)
        assert anns[0].label == "owl"
        assert anns[0].confidence == 0.9

    def test_min_confidence_filters(self) -> None:
        rows = [
            {"start": 0, "stop": 1, "prediction": "a", "confidence": 0.2},
            {"start": 1, "stop": 2, "prediction": "b", "confidence": 0.8},
        ]
        anns = predictions_to_annotations(rows, min_confidence=0.5)
        assert [a.label for a in anns] == ["b"]

    def test_rows_without_confidence_are_kept(self) -> None:
        rows = [{"start": 0, "stop": 1, "prediction": "a"}]
        anns = predictions_to_annotations(rows, min_confidence=0.9)
        assert len(anns) == 1

    def test_exclude_labels(self) -> None:
        rows = [
            {"start": 0, "stop": 1, "prediction": "background"},
            {"start": 1, "stop": 2, "prediction": "frog"},
        ]
        anns = predictions_to_annotations(rows, exclude_labels=["background"])
        assert [a.label for a in anns] == ["frog"]
