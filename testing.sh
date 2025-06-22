#!/bin/bash

# Paths
CONFIG_PATH="config/loveda/sfanet.py"
INPUT_DIR="/content/drive/MyDrive/test_data"
OUTPUT_DIR="/content/drive/MyDrive/test_data/results"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through each image in the input directory
for INPUT_IMAGE in "$INPUT_DIR"/*.jpg; do
    # Get the base filename without extension
    BASENAME=$(basename "$INPUT_IMAGE" .jpeg)
    OUTPUT_IMAGE="$OUTPUT_DIR/latest_mask_20_resize/mask_new_${BASENAME}"

    # Run prediction
    echo "Processing $INPUT_IMAGE -> $OUTPUT_IMAGE"
    python prediction.py -c "$CONFIG_PATH" -i "$INPUT_IMAGE" -o "$OUTPUT_IMAGE"
done

