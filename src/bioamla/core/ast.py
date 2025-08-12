from bioamla.core.torchaudio import load_waveform_tensor, resample_waveform_tensor, split_waveform_tensor
from transformers import ASTFeatureExtractor,AutoModelForAudioClassification
import torch
import pandas as pd
from novus_pytils.files import delete_file, file_exists


def wave_file_batch_inference(wave_files : list, model : AutoModelForAudioClassification, 
                              freq : int, clip_seconds : int, overlap_seconds : int,
                              output_csv : str) -> None:

    for filepath in wave_files:
        df = segmented_wave_file_inference(filepath, model, freq, clip_seconds, overlap_seconds)
        # results = pd.concat([results, df]) #TODO this should just return a dict. pandaing should go somewhere else
        df.to_csv(output_csv, mode='a', header=False, index=False)


def segmented_wave_file_inference(filepath : str, model: AutoModelForAudioClassification, freq : int, clip_seconds : int, overlap_seconds : int) -> pd.DataFrame:
  rows = []
  waveform, orig_freq = load_waveform_tensor(filepath)
  waveform = resample_waveform_tensor(waveform, orig_freq, freq)
  waveforms = split_waveform_tensor(waveform, freq, clip_seconds, overlap_seconds)

  for waveform in waveforms:
      input_values = extract_features(waveform[0], freq)
      prediction = ast_predict(input_values, model)
      rows.append({'filepath': filepath, 'start': waveform[1], 'stop': waveform[2], 'prediction': prediction})
  return pd.DataFrame(rows, columns=['filepath', 'start', 'stop', 'prediction'])

def wav_ast_inference(wave_path : str, model_path : str, sample_rate : int):
  waveform, orig_freq = load_waveform_tensor(wave_path)
  waveform = resample_waveform_tensor(waveform, orig_freq, sample_rate)
  input_values = extract_features(waveform, sample_rate)
  model = load_pretrained_ast_model(model_path)
  return ast_predict(input_values, model)

def ast_predict(input_values, model: AutoModelForAudioClassification) -> str:
  with torch.inference_mode():
    outputs = model(input_values)

  predicted_class_idx = outputs.logits.argmax(-1).item()
  return model.config.id2label[predicted_class_idx]

def extract_features(waveform_tensor : torch.Tensor, sample_rate : int):
  waveform_tensor = waveform_tensor.squeeze().numpy()
  feature_extractor = ASTFeatureExtractor()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  inputs = feature_extractor(waveform_tensor, sampling_rate=sample_rate, padding="max_length", return_tensors="pt").to(device)
  return inputs.input_values

def load_pretrained_ast_model(model_path : str) -> AutoModelForAudioClassification:
  return AutoModelForAudioClassification.from_pretrained(model_path, device_map="auto")
