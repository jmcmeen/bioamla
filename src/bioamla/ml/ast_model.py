"""
AST Model Wrapper
=================

:class:`ASTModel` implements the unified :class:`~bioamla.ml.base.BaseAudioModel`
interface for HuggingFace Audio Spectrogram Transformer models, providing
prediction, embedding extraction, and attention-weight inspection.

PyTorch / transformers / torchaudio are optional extras (``bioamla[ml]``). They
are imported lazily so this module imports on a slim install; constructing /
using :class:`ASTModel` without them raises
:class:`~bioamla.exceptions.DependencyError`. Load / inference failures raise
:class:`~bioamla.exceptions.ModelError`.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np

from bioamla.exceptions import DependencyError, ModelError
from bioamla.ml.base import (
    BaseAudioModel,
    ModelBackend,
    ModelConfig,
    PredictionResult,
    register_model,
)

if TYPE_CHECKING:
    import torch
    from transformers import ASTFeatureExtractor


def _require_torch():
    """Import and return the torch module, or raise DependencyError."""
    try:
        import torch
    except ImportError as e:
        raise DependencyError("AST requires torch — install bioamla[ml]") from e
    return torch


def _require_transformers():
    """Import and return AST transformers symbols, or raise DependencyError."""
    try:
        from transformers import ASTFeatureExtractor, AutoModelForAudioClassification
    except ImportError as e:
        raise DependencyError("AST requires transformers — install bioamla[ml]") from e
    return ASTFeatureExtractor, AutoModelForAudioClassification


def _torchaudio_helpers():
    """Lazily import the torchaudio waveform helpers."""
    try:
        from bioamla.audio.torchaudio import (
            load_waveform_tensor,
            resample_waveform_tensor,
            split_waveform_tensor,
        )
    except ImportError as e:
        raise DependencyError("AST requires torchaudio — install bioamla[ml]") from e
    return load_waveform_tensor, resample_waveform_tensor, split_waveform_tensor


@register_model("ast")
class ASTModel(BaseAudioModel):
    """
    Audio Spectrogram Transformer model wrapper.

    Provides a unified interface for AST models from the HuggingFace
    ``transformers`` library.

    Example:
        >>> model = ASTModel()
        >>> model.load("MIT/ast-finetuned-audioset-10-10-0.4593")
        >>> results = model.predict("audio.wav")
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        """Initialize AST model."""
        super().__init__(config)
        self.feature_extractor: ASTFeatureExtractor | None = None
        self._hook_handles: list = []

    @property
    def backend(self) -> ModelBackend:
        return ModelBackend.AST

    def load(
        self, model_path: str, use_fp16: bool = False, use_compile: bool = False, **kwargs
    ) -> "ASTModel":
        """
        Load an AST model from a path or the HuggingFace Hub.

        Args:
            model_path: Path to the model or a HuggingFace model identifier.
            use_fp16: Use half-precision inference.
            use_compile: Wrap the model with ``torch.compile()``.

        Returns:
            Self, for method chaining.

        Raises:
            DependencyError: If torch / transformers are not installed.
            ModelError: If the model cannot be loaded.
        """
        torch = _require_torch()
        ASTFeatureExtractor, AutoModelForAudioClassification = _require_transformers()

        is_local = Path(model_path).exists() or model_path.startswith(("./", "../"))
        torch_dtype = torch.float16 if use_fp16 else None

        load_kwargs = {"device_map": "auto", "torch_dtype": torch_dtype}
        if is_local:
            load_kwargs["local_files_only"] = True

        try:
            self.model = AutoModelForAudioClassification.from_pretrained(model_path, **load_kwargs)
        except Exception as e:
            raise ModelError(f"Failed to load AST model from {model_path}: {e}") from e

        if use_compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        # Load feature extractor (fall back to default if config is absent).
        try:
            self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_path)
        except OSError:
            self.feature_extractor = ASTFeatureExtractor()

        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

        self.model.eval()
        return self

    def _load_waveform(
        self,
        audio: Union[str, np.ndarray, "torch.Tensor"],
        sample_rate: int | None,
    ):
        """Coerce ``audio`` to a (waveform, sample_rate, filepath) triple."""
        torch = _require_torch()
        load_waveform_tensor, _, _ = _torchaudio_helpers()

        filepath = None
        if isinstance(audio, str):
            filepath = audio
            try:
                waveform, sample_rate = load_waveform_tensor(audio)
            except Exception as e:
                raise ModelError(f"Failed to load audio {audio}: {e}") from e
        elif isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
        else:
            waveform = audio
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

        if sample_rate is None:
            sample_rate = self.config.sample_rate

        return waveform, sample_rate, filepath

    def predict(
        self,
        audio: Union[str, np.ndarray, "torch.Tensor"],
        sample_rate: int | None = None,
    ) -> list[PredictionResult]:
        """
        Run prediction on audio, returning one result per segment.

        Args:
            audio: Audio file path, numpy array, or torch tensor.
            sample_rate: Sample rate if ``audio`` is an array/tensor.

        Returns:
            List of prediction results.

        Raises:
            DependencyError: If torch / transformers / torchaudio are missing.
            ModelError: If the model is not loaded or inference fails.
        """
        torch = _require_torch()
        _, resample_waveform_tensor, split_waveform_tensor = _torchaudio_helpers()

        if self.model is None:
            raise ModelError("Model not loaded. Call load() first.")

        waveform, sample_rate, filepath = self._load_waveform(audio, sample_rate)

        if sample_rate != self.config.sample_rate:
            waveform = resample_waveform_tensor(waveform, sample_rate, self.config.sample_rate)

        clip_samples = int(self.config.clip_duration * self.config.sample_rate)

        if waveform.shape[1] <= clip_samples:
            segments = [(waveform, 0, waveform.shape[1])]
        else:
            segments = split_waveform_tensor(
                waveform,
                self.config.sample_rate,
                int(self.config.clip_duration),
                int(self.config.overlap),
            )

        results = []
        for segment, start_sample, end_sample in segments:
            waveform_np = segment.squeeze().numpy()
            inputs = self.feature_extractor(
                waveform_np,
                sampling_rate=self.config.sample_rate,
                padding="max_length",
                return_tensors="pt",
            )

            input_values = inputs.input_values.to(self.device)
            if self.config.use_fp16 and self.device.type == "cuda":
                input_values = input_values.half()

            try:
                with torch.inference_mode():
                    outputs = self.model(input_values)
            except Exception as e:
                raise ModelError(f"AST inference failed: {e}") from e

            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)

            top_k = min(self.config.top_k, len(probs))
            top_probs, top_indices = torch.topk(probs, top_k)

            for prob, idx in zip(top_probs, top_indices):
                label = self.id2label[idx.item()]
                confidence = prob.item()
                if confidence >= self.config.min_confidence:
                    results.append(
                        PredictionResult(
                            label=label,
                            confidence=confidence,
                            logits=logits.cpu().numpy() if top_k == 1 else None,
                            start_time=start_sample / self.config.sample_rate,
                            end_time=end_sample / self.config.sample_rate,
                            filepath=filepath,
                        )
                    )

        return results

    def extract_embeddings(
        self,
        audio: Union[str, np.ndarray, "torch.Tensor"],
        sample_rate: int | None = None,
        layer: str | None = None,
    ) -> np.ndarray:
        """
        Extract embeddings from audio (mean-pooled hidden state).

        Args:
            audio: Audio file path, numpy array, or torch tensor.
            sample_rate: Sample rate if ``audio`` is an array/tensor.
            layer: Layer to extract from (``last_hidden_state``, ``layer_<n>``,
                or default = mean of the last hidden state).

        Returns:
            Embedding vectors as a numpy array.

        Raises:
            DependencyError: If torch / transformers / torchaudio are missing.
            ModelError: If the model is not loaded or inference fails.
        """
        torch = _require_torch()
        _, resample_waveform_tensor, _ = _torchaudio_helpers()

        if self.model is None:
            raise ModelError("Model not loaded. Call load() first.")

        waveform, sample_rate, _ = self._load_waveform(audio, sample_rate)

        if sample_rate != self.config.sample_rate:
            waveform = resample_waveform_tensor(waveform, sample_rate, self.config.sample_rate)

        waveform_np = waveform.squeeze().numpy()
        inputs = self.feature_extractor(
            waveform_np,
            sampling_rate=self.config.sample_rate,
            padding="max_length",
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(self.device)

        try:
            with torch.inference_mode():
                outputs = self.model(input_values, output_hidden_states=True)
        except Exception as e:
            raise ModelError(f"AST embedding extraction failed: {e}") from e

        if layer == "last_hidden_state":
            embeddings = outputs.hidden_states[-1].mean(dim=1)
        elif layer and layer.startswith("layer_"):
            layer_idx = int(layer.split("_")[1])
            embeddings = outputs.hidden_states[layer_idx].mean(dim=1)
        else:
            embeddings = outputs.hidden_states[-1].mean(dim=1)

        return embeddings.cpu().numpy()

    def _get_dummy_input(self) -> "torch.Tensor":
        """Get a dummy feature-extractor output for model export."""
        dummy_audio = np.random.randn(self.config.sample_rate).astype(np.float32)
        inputs = self.feature_extractor(
            dummy_audio,
            sampling_rate=self.config.sample_rate,
            padding="max_length",
            return_tensors="pt",
        )
        return inputs.input_values.to(self.device)

    def get_attention_weights(
        self,
        audio: Union[str, np.ndarray, "torch.Tensor"],
        sample_rate: int | None = None,
    ) -> list[np.ndarray]:
        """
        Get per-layer attention weight matrices for the audio.

        Args:
            audio: Audio file path, numpy array, or torch tensor.
            sample_rate: Sample rate if ``audio`` is an array/tensor.

        Returns:
            List of attention weight matrices, one per layer.

        Raises:
            DependencyError: If torch / transformers / torchaudio are missing.
            ModelError: If the model is not loaded or inference fails.
        """
        torch = _require_torch()
        _, resample_waveform_tensor, _ = _torchaudio_helpers()

        if self.model is None:
            raise ModelError("Model not loaded. Call load() first.")

        waveform, sample_rate, _ = self._load_waveform(audio, sample_rate)

        if sample_rate != self.config.sample_rate:
            waveform = resample_waveform_tensor(waveform, sample_rate, self.config.sample_rate)

        waveform_np = waveform.squeeze().numpy()
        inputs = self.feature_extractor(
            waveform_np,
            sampling_rate=self.config.sample_rate,
            padding="max_length",
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(self.device)

        try:
            with torch.inference_mode():
                outputs = self.model(input_values, output_attentions=True)
        except Exception as e:
            raise ModelError(f"AST attention extraction failed: {e}") from e

        return [attn.cpu().numpy() for attn in outputs.attentions]
