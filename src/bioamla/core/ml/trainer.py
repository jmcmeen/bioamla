"""
Model Training Pipeline
=======================

Custom CNN training pipeline for transfer learning on audio data.
Supports training OpenSoundscape-style models with various backbone
architectures.
"""

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, Dataset

from bioamla.core.files import TextFile
from bioamla.core.ml.base import BaseAudioModel, ModelConfig


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Data parameters
    train_dir: str = ""
    val_dir: Optional[str] = None
    output_dir: str = "./output"
    class_names: List[str] = field(default_factory=list)

    # Model parameters
    architecture: str = "resnet18"
    pretrained: bool = True
    freeze_backbone_epochs: int = 0  # Epochs to keep backbone frozen

    # Training parameters
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # adam, adamw, sgd
    scheduler: str = "cosine"  # cosine, plateau, step, none

    # Audio parameters
    sample_rate: int = 16000
    clip_duration: float = 3.0

    # Augmentation
    augment: bool = True
    mixup_alpha: float = 0.0  # 0 = disabled

    # Hardware
    device: Optional[str] = None
    num_workers: int = 4
    use_fp16: bool = False

    # Logging
    log_every: int = 10
    save_every: int = 1  # Save checkpoint every N epochs
    early_stopping_patience: int = 5


@dataclass
class TrainingMetrics:
    """Training metrics for tracking."""

    epoch: int = 0
    train_loss: float = 0.0
    train_accuracy: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    learning_rate: float = 0.0
    epoch_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "learning_rate": self.learning_rate,
            "epoch_time": self.epoch_time,
        }


class SpectrogramDataset(Dataset):
    """Dataset for spectrogram-based training."""

    def __init__(
        self,
        data_dir: str,
        class_names: List[str],
        sample_rate: int = 16000,
        clip_duration: float = 3.0,
        transform: Optional[Callable] = None,
        augment: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory with class subdirectories containing audio files.
            class_names: List of class names (subdirectory names).
            sample_rate: Target sample rate.
            clip_duration: Clip duration in seconds.
            transform: Optional spectrogram transform.
            augment: Apply augmentation.
        """
        self.data_dir = Path(data_dir)
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.transform = transform
        self.augment = augment

        # Collect samples
        self.samples: List[Tuple[str, int]] = []
        self._collect_samples()

    def _collect_samples(self) -> None:
        """Collect audio file paths and labels."""
        audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

        for class_name in self.class_names:
            class_dir = self.data_dir / class_name

            if not class_dir.exists():
                continue

            class_idx = self.class_to_idx[class_name]

            for audio_file in class_dir.iterdir():
                if audio_file.suffix.lower() in audio_extensions:
                    self.samples.append((str(audio_file), class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        from bioamla.core.audio.torchaudio import load_waveform_tensor, resample_waveform_tensor

        filepath, label = self.samples[idx]

        # Load audio
        waveform, sr = load_waveform_tensor(filepath)

        if sr != self.sample_rate:
            waveform = resample_waveform_tensor(waveform, sr, self.sample_rate)

        # Truncate or pad
        target_samples = int(self.clip_duration * self.sample_rate)
        if waveform.shape[1] > target_samples:
            # Random crop during training
            if self.augment:
                start = np.random.randint(0, waveform.shape[1] - target_samples)
            else:
                start = 0
            waveform = waveform[:, start:start + target_samples]
        elif waveform.shape[1] < target_samples:
            padding = target_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Compute spectrogram
        spectrogram = self._compute_spectrogram(waveform)

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram.squeeze(0), label

    def _compute_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram."""
        import torchaudio.transforms as T

        mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
        )

        mel_spec = mel_transform(waveform)

        db_transform = T.AmplitudeToDB(stype="power", top_db=80)
        mel_spec_db = db_transform(mel_spec)

        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

        # Resize to standard size
        mel_spec_db = torch.nn.functional.interpolate(
            mel_spec_db.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return mel_spec_db


class ModelTrainer:
    """
    Trainer for audio classification models.

    Supports transfer learning with backbone freezing,
    mixed precision training, and various optimization strategies.

    Example:
        >>> config = TrainingConfig(
        ...     train_dir="./data/train",
        ...     val_dir="./data/val",
        ...     class_names=["bird", "frog", "insect"],
        ...     num_epochs=20,
        ... )
        >>> trainer = ModelTrainer(config)
        >>> model = trainer.train()
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.

        Args:
            config: Training configuration.
        """
        self.config = config
        self.device = torch.device(
            config.device if config.device else
            ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None

        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

        self.history: List[TrainingMetrics] = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def setup(self, model: Optional[nn.Module] = None) -> None:
        """
        Setup training components.

        Args:
            model: Optional pre-created model. If None, creates from config.
        """
        # Create model if not provided
        if model is None:
            from bioamla.ml.opensoundscape import SpectrogramCNN

            self.model = SpectrogramCNN(
                num_classes=len(self.config.class_names),
                architecture=self.config.architecture,
                pretrained=self.config.pretrained,
            )
        else:
            self.model = model

        self.model.to(self.device)

        # Create datasets
        train_dataset = SpectrogramDataset(
            data_dir=self.config.train_dir,
            class_names=self.config.class_names,
            sample_rate=self.config.sample_rate,
            clip_duration=self.config.clip_duration,
            augment=self.config.augment,
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        if self.config.val_dir:
            val_dataset = SpectrogramDataset(
                data_dir=self.config.val_dir,
                class_names=self.config.class_names,
                sample_rate=self.config.sample_rate,
                clip_duration=self.config.clip_duration,
                augment=False,
            )

            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )

        # Setup optimizer
        self._setup_optimizer()

        # Setup scheduler
        self._setup_scheduler()

        # Setup mixed precision
        if self.config.use_fp16 and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()

    def _setup_optimizer(self) -> None:
        """Setup optimizer based on config."""
        params = self.model.parameters()

        if self.config.optimizer == "adam":
            self.optimizer = Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adamw":
            self.optimizer = AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "sgd":
            self.optimizer = SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        if self.config.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
            )
        elif self.config.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=3,
            )
        elif self.config.scheduler == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=5,
                gamma=0.5,
            )
        elif self.config.scheduler != "none":
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

    def train(
        self,
        progress_callback: Optional[Callable[[int, int, TrainingMetrics], None]] = None,
    ) -> nn.Module:
        """
        Run training loop.

        Args:
            progress_callback: Optional callback(epoch, total_epochs, metrics).

        Returns:
            Trained model.
        """
        if self.model is None:
            self.setup()

        criterion = nn.CrossEntropyLoss()
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()

            # Handle backbone freezing
            if epoch < self.config.freeze_backbone_epochs:
                if hasattr(self.model, "freeze_backbone"):
                    self.model.freeze_backbone()
            else:
                if hasattr(self.model, "unfreeze_backbone"):
                    self.model.unfreeze_backbone()

            # Training phase
            train_loss, train_acc = self._train_epoch(criterion)

            # Validation phase
            val_loss, val_acc = 0.0, 0.0
            if self.val_loader:
                val_loss, val_acc = self._validate_epoch(criterion)

            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss if self.val_loader else train_loss)
                else:
                    self.scheduler.step()

            # Record metrics
            current_lr = self.optimizer.param_groups[0]["lr"]
            metrics = TrainingMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
                learning_rate=current_lr,
                epoch_time=time.time() - epoch_start,
            )
            self.history.append(metrics)

            # Progress callback
            if progress_callback:
                progress_callback(epoch + 1, self.config.num_epochs, metrics)

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(output_dir / f"checkpoint_epoch_{epoch + 1}.pt")

            # Check for best model
            val_metric = val_loss if self.val_loader else train_loss
            if val_metric < self.best_val_loss:
                self.best_val_loss = val_metric
                self._save_checkpoint(output_dir / "best_model.pt")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Save final model
        self._save_checkpoint(output_dir / "final_model.pt")
        self._save_training_history(output_dir / "training_history.json")

        return self.model

    def _train_epoch(self, criterion: nn.Module) -> Tuple[float, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def _validate_epoch(self, criterion: nn.Module) -> Tuple[float, float]:
        """Run validation epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.inference_mode():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def _save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": len(self.history),
            "best_val_loss": self.best_val_loss,
            "config": {
                "architecture": self.config.architecture,
                "num_classes": len(self.config.class_names),
                "sample_rate": self.config.sample_rate,
                "clip_duration": self.config.clip_duration,
            },
            "id2label": {i: name for i, name in enumerate(self.config.class_names)},
            "label2id": {name: i for i, name in enumerate(self.config.class_names)},
        }

        torch.save(checkpoint, path)

    def _save_training_history(self, path: Path) -> None:
        """Save training history to JSON."""
        history_data = [m.to_dict() for m in self.history]
        with TextFile(path, mode="w") as f:
            json.dump(history_data, f.handle, indent=2)


def train_model(
    train_dir: str,
    class_names: List[str],
    output_dir: str = "./output",
    val_dir: Optional[str] = None,
    architecture: str = "resnet18",
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    pretrained: bool = True,
    freeze_epochs: int = 0,
    verbose: bool = True,
) -> str:
    """
    Train a model using transfer learning.

    Convenience function for training an OpenSoundscape-style model.

    Args:
        train_dir: Directory containing class subdirectories with audio.
        class_names: List of class names (subdirectory names).
        output_dir: Directory to save outputs.
        val_dir: Optional validation directory.
        architecture: Model architecture ("resnet18", "resnet50").
        num_epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.
        pretrained: Use ImageNet pretrained weights.
        freeze_epochs: Epochs to keep backbone frozen.
        verbose: Print progress.

    Returns:
        Path to the best model checkpoint.
    """
    config = TrainingConfig(
        train_dir=train_dir,
        val_dir=val_dir,
        output_dir=output_dir,
        class_names=class_names,
        architecture=architecture,
        pretrained=pretrained,
        freeze_backbone_epochs=freeze_epochs,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
    )

    trainer = ModelTrainer(config)
    trainer.setup()

    def progress(epoch: int, total: int, metrics: TrainingMetrics):
        if verbose:
            val_info = ""
            if metrics.val_loss > 0:
                val_info = f", val_loss: {metrics.val_loss:.4f}, val_acc: {metrics.val_accuracy:.4f}"
            print(
                f"Epoch {epoch}/{total} - "
                f"loss: {metrics.train_loss:.4f}, acc: {metrics.train_accuracy:.4f}"
                f"{val_info}"
            )

    trainer.train(progress_callback=progress)

    return str(Path(output_dir) / "best_model.pt")
