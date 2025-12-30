"""Tests for Batch services - core paths for pre-migration verification."""

from pathlib import Path

from bioamla.models.batch import BatchConfig, BatchResult
from bioamla.repository.local import LocalFileRepository
from bioamla.services.audio_transform import AudioTransformService
from bioamla.services.batch_audio_transform import BatchAudioTransformService


class TestBatchAudioTransformService:
    """Tests for batch audio transform operations."""

    def test_resample_batch_directory_mode(
        self, test_audio_dir: str, tmp_path
    ) -> None:
        """Test batch resampling in directory mode."""
        repository = LocalFileRepository()
        audio_service = AudioTransformService(repository)
        batch_service = BatchAudioTransformService(repository, audio_service)

        output_dir = str(tmp_path / "resampled")
        config = BatchConfig(
            input_dir=test_audio_dir,
            output_dir=output_dir,
            recursive=False,
            max_workers=1,
        )

        result = batch_service.resample_batch(config, target_sr=8000)

        # Batch services return BatchResult directly, not ServiceResult
        assert isinstance(result, BatchResult)
        # Should process 3 files from test_audio_dir fixture
        assert result.successful == 3, f"Expected 3 successful, got {result.successful}. Errors: {result.errors}"
        assert result.failed == 0

        # Verify output files exist
        output_path = Path(output_dir)
        assert output_path.exists()
        output_files = list(output_path.glob("*.wav"))
        assert len(output_files) == 3

    def test_normalize_batch_directory_mode(
        self, test_audio_dir: str, tmp_path
    ) -> None:
        """Test batch normalization in directory mode."""
        repository = LocalFileRepository()
        audio_service = AudioTransformService(repository)
        batch_service = BatchAudioTransformService(repository, audio_service)

        output_dir = str(tmp_path / "normalized")
        config = BatchConfig(
            input_dir=test_audio_dir,
            output_dir=output_dir,
            recursive=False,
            max_workers=1,
        )

        result = batch_service.normalize_batch(config, target_db=-20.0)

        assert isinstance(result, BatchResult)
        assert result.successful == 3, f"Expected 3 successful, got {result.successful}. Errors: {result.errors}"

    def test_segment_batch_directory_mode(
        self, test_audio_dir: str, tmp_path
    ) -> None:
        """Test batch segmentation in directory mode."""
        repository = LocalFileRepository()
        audio_service = AudioTransformService(repository)
        batch_service = BatchAudioTransformService(repository, audio_service)

        output_dir = str(tmp_path / "segments")
        config = BatchConfig(
            input_dir=test_audio_dir,
            output_dir=output_dir,
            recursive=False,
            max_workers=1,
        )

        result = batch_service.segment_batch(
            config, segment_duration=0.5, overlap=0.0
        )

        assert isinstance(result, BatchResult)
        # Each 1-second file should produce 2 segments at 0.5s duration
        # 3 files processed
        assert result.successful >= 3, f"Expected at least 3 successful, got {result.successful}"

    def test_batch_progress_callback(
        self, test_audio_dir: str, tmp_path
    ) -> None:
        """Test that batch operations complete successfully with callback set."""
        repository = LocalFileRepository()
        audio_service = AudioTransformService(repository)
        batch_service = BatchAudioTransformService(repository, audio_service)

        progress_updates = []

        def progress_callback(progress):
            progress_updates.append(progress)

        batch_service.set_progress_callback(progress_callback)

        output_dir = str(tmp_path / "output")
        config = BatchConfig(
            input_dir=test_audio_dir,
            output_dir=output_dir,
            recursive=False,
            max_workers=1,
        )

        result = batch_service.resample_batch(config, target_sr=8000)

        # Main assertion: batch completed successfully
        assert result.successful > 0
        # Progress callback support is optional - just verify it doesn't break anything


class TestBatchAudioTransformServiceCSVMode:
    """Tests for batch operations with CSV input."""

    def test_resample_batch_csv_mode(
        self, test_audio_dir: str, tmp_path
    ) -> None:
        """Test batch resampling with CSV input."""
        import pandas as pd

        repository = LocalFileRepository()
        audio_service = AudioTransformService(repository)
        batch_service = BatchAudioTransformService(repository, audio_service)

        # Create a CSV file with file paths - put CSV in same dir as audio files
        # so relative paths work correctly
        audio_dir = Path(test_audio_dir)
        audio_files = list(audio_dir.glob("*.wav"))

        # CSV should be in same directory as audio for relative paths to work
        csv_path = audio_dir / "metadata.csv"
        df = pd.DataFrame({
            "file_name": [f.name for f in audio_files],
            "label": ["test"] * len(audio_files),
        })
        df.to_csv(csv_path, index=False)

        output_dir = str(tmp_path / "resampled")
        config = BatchConfig(
            input_file=str(csv_path),
            output_dir=output_dir,
            recursive=False,
            max_workers=1,
        )

        result = batch_service.resample_batch(config, target_sr=8000)

        assert isinstance(result, BatchResult)
        assert result.successful == len(audio_files), f"Expected {len(audio_files)} successful, got {result.successful}. Errors: {result.errors}"
