# Methods for working with wav files
import torch
import torchaudio
from torchaudio.transforms import Resample
from novus_pytils.files import get_files_by_extension

def get_wav_info(filepath : str):
    return torchaudio.info(filepath)

def get_wav_files(directory : str) -> list:
    return get_files_by_extension(directory, ['.wav'])

def get_wavefile_shape(wavefile_path : str):
    waveform, _ = torchaudio.load(wavefile_path)
    return waveform.shape

def get_wavefile_sample_rate(wavefile_path : str):
    _, sample_rate = torchaudio.load(wavefile_path)
    return sample_rate

def load_waveform_tensor(filepath : str):
  waveform, sample_rate = torchaudio.load(filepath)
  return (waveform, sample_rate)

def split_waveform_tensor(waveform_tensor : torch.Tensor, freq : int, clip_seconds : int, overlap_seconds : int): 
  segment_size = int(clip_seconds * freq)
  step_size = int((clip_seconds - overlap_seconds) * freq)

  segments = []
  start = 0
  while start + segment_size <= waveform_tensor.shape[1]:
    segment = waveform_tensor[:, start:start+segment_size]
    segments.append((segment, start, start+segment_size))
    start += step_size
  return segments

def resample_waveform_tensor(waveform_tensor : torch.Tensor, orig_freq : int, new_freq : int):
  resampler = Resample(orig_freq=orig_freq, new_freq=new_freq)
  waveform_tensor = resampler(waveform_tensor)
  return waveform_tensor
