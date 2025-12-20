"""
AST Model Wrapper
=================

Audio Spectrogram Transformer (AST) model implementation using the
unified BaseAudioModel interface. Wraps the existing bioamla.ast module
and HuggingFace transformers.
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from transformers import ASTFeatureExtractor, AutoModelForAudioClassification

from bioamla.core.audio.torchaudio import (
    load_waveform_tensor,
    resample_waveform_tensor,
    split_waveform_tensor,
)
from bioamla.core.ml.base import (
    BaseAudioModel,
    ModelBackend,
    ModelConfig,
    PredictionResult,
    register_model,
)


@register_model("ast")
class ASTModel(BaseAudioModel):
    """
    Audio Spectrogram Transformer model wrapper.

    This class provides a unified interface for AST models from
    HuggingFace transformers library.

    Example:
        >>> model = ASTModel()
        >>> model.load("MIT/ast-finetuned-audioset-10-10-0.4593")
        >>> results = model.predict("audio.wav")
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize AST model."""
        super().__init__(config)
        self.feature_extractor: Optional[ASTFeatureExtractor] = None
        self._hook_handles = []

    @property
    def backend(self) -> ModelBackend:
        return ModelBackend.AST

    def load(
        self, model_path: str, use_fp16: bool = False, use_compile: bool = False, **kwargs
    ) -> "ASTModel":
        """
        Load AST model from path or HuggingFace Hub.

        Args:
            model_path: Path to model or HuggingFace model identifier.
            use_fp16: Use half-precision inference.
            use_compile: Use torch.compile() for optimization.

        Returns:
            Self for method chaining.
        """
        is_local = Path(model_path).exists() or model_path.startswith(("./", "../"))
        torch_dtype = torch.float16 if use_fp16 else None

        load_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch_dtype,
        }

        if is_local:
            load_kwargs["local_files_only"] = True

        self.model = AutoModelForAudioClassification.from_pretrained(model_path, **load_kwargs)

        if use_compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        # Load feature extractor
        try:
            self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_path)
        except OSError:
            # Model doesn't have preprocessor_config.json, use default
            self.feature_extractor = ASTFeatureExtractor()

        # Setup label mappings
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

        self.model.eval()
        return self

    def predict(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
    ) -> List[PredictionResult]:
        """
        Run prediction on audio.

        Args:
            audio: Audio file path, numpy array, or torch tensor.
            sample_rate: Sample rate if audio is array/tensor.

        Returns:
            List of prediction results (one per segment).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Load audio if filepath
        filepath = None
        if isinstance(audio, str):
            filepath = audio
            waveform, orig_sr = load_waveform_tensor(audio)
            sample_rate = orig_sr
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

        # Resample if needed
        if sample_rate != self.config.sample_rate:
            waveform = resample_waveform_tensor(waveform, sample_rate, self.config.sample_rate)

        # Split into segments
        clip_samples = int(self.config.clip_duration * self.config.sample_rate)
        overlap_samples = int(self.config.overlap * self.config.sample_rate)

        if waveform.shape[1] <= clip_samples:
            # Single segment
            segments = [(waveform, 0, waveform.shape[1])]
        else:
            segments = split_waveform_tensor(
                waveform,
                self.config.sample_rate,
                int(self.config.clip_duration),
                int(self.config.overlap),
            )

        # Process segments
        results = []
        for segment, start_sample, end_sample in segments:
            # Extract features
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

            # Run inference
            with torch.inference_mode():
                outputs = self.model(input_values)

            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)

            # Get top-k predictions
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
        audio: Union[str, np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
        layer: Optional[str] = None,
    ) -> np.ndarray:
        """
        Extract embeddings from audio.

        For AST models, embeddings are extracted from the pooler output
        or a specified hidden layer.

        Args:
            audio: Audio file path, numpy array, or torch tensor.
            sample_rate: Sample rate if audio is array/tensor.
            layer: Layer to extract from (default: pooler_output).

        Returns:
            Embedding vectors as numpy array.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Load audio if filepath
        if isinstance(audio, str):
            waveform, orig_sr = load_waveform_tensor(audio)
            sample_rate = orig_sr
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

        # Resample if needed
        if sample_rate != self.config.sample_rate:
            waveform = resample_waveform_tensor(waveform, sample_rate, self.config.sample_rate)

        # Extract features
        waveform_np = waveform.squeeze().numpy()
        inputs = self.feature_extractor(
            waveform_np,
            sampling_rate=self.config.sample_rate,
            padding="max_length",
            return_tensors="pt",
        )

        input_values = inputs.input_values.to(self.device)

        # Run inference with hidden states
        with torch.inference_mode():
            outputs = self.model(input_values, output_hidden_states=True)

        if layer == "last_hidden_state":
            # Use mean of last hidden state
            embeddings = outputs.hidden_states[-1].mean(dim=1)
        elif layer and layer.startswith("layer_"):
            # Extract specific layer
            layer_idx = int(layer.split("_")[1])
            embeddings = outputs.hidden_states[layer_idx].mean(dim=1)
        else:
            # Use the hidden state before the classifier (second to last if available)
            # For AST, use mean pooling of last hidden state
            embeddings = outputs.hidden_states[-1].mean(dim=1)

        return embeddings.cpu().numpy()

    def _get_dummy_input(self) -> torch.Tensor:
        """Get dummy input for model export."""
        # AST expects spectrogram features, not raw audio
        # Create dummy feature extractor output
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
        audio: Union[str, np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Get attention weights from the model.

        Args:
            audio: Audio file path, numpy array, or torch tensor.
            sample_rate: Sample rate if audio is array/tensor.

        Returns:
            List of attention weight matrices per layer.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Load audio if filepath
        if isinstance(audio, str):
            waveform, orig_sr = load_waveform_tensor(audio)
            sample_rate = orig_sr
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

        with torch.inference_mode():
            outputs = self.model(input_values, output_attentions=True)

        return [attn.cpu().numpy() for attn in outputs.attentions]
