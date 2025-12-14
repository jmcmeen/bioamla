#!/bin/bash

# Example workflow: Download iNaturalist audio and convert to WAV format

# Step 1: Download audio observations from iNaturalist
bioamla inat-audio ./dev_data/test1 \
    --taxon-ids "24268, 65982, 23930, 24263, 65979, 66002, 66012, 60341, 64968, 64977, 24256" \
    --quality-grade research \
    --obs-per-taxon 2

# Step 2: Convert all audio files in the dataset to WAV format
bioamla convert-audio ./dev_data/test1 wav
