#!/bin/bash

# Example workflow: Download iNaturalist audio and convert to WAV format

# Step 1: Download audio observations from iNaturalist
bioamla inat-audio ./frogs_dataset \
    --taxon-ids "24268, 65982, 23930, 24263, 65979, 66002, 66012, 60341, 64968, 64977, 24256" \
    --quality-grade research \
    --obs-per-taxon 6

# Step 2: Convert all audio files in the dataset to WAV format
bioamla convert-audio ./frogs_dataset wav

# Step 3: Fine-tune an audio spectrogram transformer (AST) model on the downloaded dataset
bioamla ast-finetune --training-dir ./frogs_out --train-dataset ./frogs_dataset --num-train-epochs 2

# Step 4: Download a test dataset
bioamla download https://www.bioamla.org/datasets/scp_small.zip .

# Step 5: Unzip the test dataset
bioamla unzip scp_small.zip .

# Step 6: Run batch inference on the test dataset using the fine-tuned model
bioamla ast-batch-inference ./scp_small --model-path ./frogs_out/best_model
