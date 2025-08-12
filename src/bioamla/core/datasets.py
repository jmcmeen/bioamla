
def validate_metadata(audio_folder_path, metadata_csv_filename='metadata.csv'):
    import pandas as pd
    import os
    metadata_df = pd.read_csv(os.path.join(audio_folder_path,metadata_csv_filename))

    #check that the audio folder contains the same number of files as the metadata.csv file
    num_audio_files = count_audio_files(audio_folder_path)
    num_metadata_files = len(metadata_df)
    if num_audio_files != num_metadata_files:
        raise ValueError(f"The number of audio files in the audio folder ({num_audio_files}) does not match the number of files in the metadata.csv file ({num_metadata_files})")

    #check that all audio files are in metadata
    audio_files = get_audio_files(audio_folder_path)
    for audio_file in audio_files:
        if audio_file not in metadata_df['filename'].tolist():
            raise ValueError(f"The audio file {audio_file} is not in the metadata.csv file")

    return True

def load_local_dataset(audio_folder_path):
    from datasets import load_dataset
    from novus_pytils.files import directory_exists
    if not directory_exists(audio_folder_path):
        raise ValueError(f"The audio folder {audio_folder_path} does not exist")
    if count_audio_files(audio_folder_path) == 0:
        raise ValueError(f"The audio folder {audio_folder_path} is empty")
    
    dataset = load_dataset(audio_folder_path)
    
    
    return dataset