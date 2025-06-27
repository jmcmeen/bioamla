from modules.audio_torch import *
import os
if __name__ == '__main__':
    src_dir = './data/audio/wav copy'
    dataset_dir = './data/audio/wav copy small'
    
    copy_dataset(src_dir, dataset_dir)
    resample_wave_files(dataset_dir, sample_rate=16000)
    bandpass_wave_files(dataset_dir, low_f=300, high_f=4000, order=12)
    split_wave_files(dataset_dir, clip_seconds=1, clip_overlap=0)

    df = extract_partitioned_dataset(dataset_dir, ['.wav'])
    df.to_csv(os.path.join(dataset_dir, "out.csv"), index=None)

    # df = extract_partitioned_dataset('C:\\riffytest\\scp_small', [".wav"])
    # df = transform_to_onehot(df)
    # df.to_csv('C:\\riffytest\\scp_small.csv', index=False)
    # print(df)
