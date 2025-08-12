import click
from novus_pytils.files import get_files_by_extension, file_exists
from novus_pytils.config.yaml import load_yaml
from bioamla.core.ast import load_pretrained_ast_model, wave_file_batch_inference
import torch
import sys
import time
import pandas as pd
import os

@click.command()
@click.argument('config_filepath')
def main(config_filepath : str):
    print ("Loading config file: " + config_filepath)
    config = load_yaml(config_filepath)

    output_csv = os.path.join(config['directory'], config['output_csv'])
    print("Output csv: " + output_csv)

    wave_files = get_files_by_extension(config["directory"], ['.wav'])

    if(len(wave_files) == 0):
        print("No wave files found in directory: " + config["directory"])
        return
    else:
        print("Found " + str(len(wave_files)) + " wave files in directory: " + config["directory"])

    if config['restart']:
        print("Restart: " + str(config['restart']))

            #if file exists, read file names from file and remove from wave files
        if file_exists(output_csv):
            print("file exists: " + output_csv)
            df = pd.read_csv(output_csv)
            #filenames exist more than once get the unique ones
            processed_files = set(df['filepath'])
            print("Found " + str(len(processed_files)) + " processed files")

            print("Removing processed files from wave files")
            # Todo use sets
            for filepath in processed_files:
                wave_files.remove(filepath)

            print("Found " + str(len(wave_files)) + " wave files left to process")

            if len(wave_files) == 0:
                print("No wave files left to process")
                return
        else:
            print("creating new file: " + output_csv)
            results = pd.DataFrame(columns=['filepath', 'start', 'stop', 'prediction'])
            results.to_csv(output_csv, header=True, index=False)

            
    else:
        print("creating new file: " + output_csv)
        results = pd.DataFrame(columns=['filepath', 'start', 'stop', 'prediction'])
        results.to_csv(output_csv, header=True, index=False)

    print("Loading model: " + config["model"])
    model = load_pretrained_ast_model(config["model"])
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device: " + device)
    model.to(device)

    # start timercurrent
    start_time = time.time()
    #format start time
    time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print("Start batch inference at " + time_string)
    wave_file_batch_inference(wave_files=wave_files, 
                              model=model,
                              freq=config["resample_freq"], 
                              clip_seconds=config["clip_seconds"], 
                              overlap_seconds=config["overlap_seconds"],
                              output_csv=output_csv)
    
    # end timer
    end_time = time.time()
    time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    print("End batch inference at " + time_string)
    print("Elapsed time: " + str(end_time - start_time))

if __name__ == '__main__':
    main()