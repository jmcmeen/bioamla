import io
from pathlib import Path  
from bioamla.core.exceptions import (
    NoModelLoadedError,
    UnsupportAudioFormatError
    )
from bioamla.core.models.responses import AudioClassificationResponse
from bioamla.core.models.config import DefaultConfig
from bioamla.core.helpers.wav_torch import load_audio_from_bytes
from bioamla.core.helpers.transformers import process_audio_with_pipeline

async def classify_audio(model, audio_pipeline,file, top_k = 5):
    """Classify an uploaded audio file."""
    import time
    start_time = time.time()
    
    if model is None or audio_pipeline is None:
        raise NoModelLoadedError()
    
    # Validate file type
    allowed_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        # raise HTTPException(
        #     status_code=400,
        #     detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        # )
        raise UnsupportAudioFormatError()
    
    try:
        # Read audio file
        audio_bytes = await file.read()
        
        # Load and preprocess audio
        audio_array, sample_rate = load_audio_from_bytes(audio_bytes)
        
        # Calculate duration
        duration = len(audio_array) / sample_rate
        
        # Check audio length
        if duration > DefaultConfig.MAX_AUDIO_LENGTH:
            # Trim audio to max length
            max_samples = int(DefaultConfig.MAX_AUDIO_LENGTH * sample_rate)
            audio_array = audio_array[:max_samples]
            duration = DefaultConfig.MAX_AUDIO_LENGTH
        
        # Process audio
        predictions = process_audio_with_pipeline(audio_array, sample_rate, top_k)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return AudioClassificationResponse(
            success=True,
            predictions=predictions,
            audio_duration=duration,
            sample_rate=sample_rate,
            model_used=DefaultConfig.MODEL_NAME,
            processing_time=processing_time
        )
    except Exception as e:
        # TODO Catch specific exceptions
        raise
    
