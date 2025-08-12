
import numpy as np
from bioamla.core.models.responses import PredictionResult

def process_audio_with_pipeline(audio_pipeline, audio_array: np.ndarray, sample_rate: int, top_k: int = 5):
    """Process audio using the transformer pipeline."""
    try:
        # Run inference
        results = audio_pipeline(
            audio_array,
            sampling_rate=sample_rate,
            top_k=top_k
        )
        
        # Format results
        predictions = []
        for i, pred in enumerate(results):
            predictions.append(
                PredictionResult(
                    label=pred['label'],
                    score=float(pred['score']),
                    rank=i + 1
                )
            )
        
        return predictions
    except Exception:
        # TODO Catch specific exceptions
        raise