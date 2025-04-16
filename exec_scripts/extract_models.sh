#!/bin/bash

# Define base path
SRC_BASE="/mounted_volume/deepreach/runs/mulob/ConveyorND/decomposedND"
DST_BASE="/mounted_volume/deepreach/value_fns/Conveyor/models/reach_only"

# List of target directories
DIRS=(
    "reach_only_4D_p2"
    "reach_only_8D_p5"
    "reach_only_16D_p4"
    "reach_only_32D_p3"
    "reach_only_64D_p5"
    "reach_only_128D_p5"
)

for dir in "${DIRS[@]}"; do
    SRC_DIR="$SRC_BASE/$dir"
    DST_DIR="$DST_BASE/$dir"

    echo "Copying $dir..."

    # Create full directory structure
    mkdir -p "$DST_DIR"
    
    # Copy top-level files (excluding training/)
    find "$SRC_DIR" -maxdepth 1 -type f -exec cp {} "$DST_DIR/" \;

    # Copy everything except 'training' directory
    find "$SRC_DIR" -mindepth 1 -maxdepth 1 ! -name "training" -exec cp -r {} "$DST_DIR/" \;

    # Create empty directories for checkpoints and summaries
    mkdir -p "$DST_DIR/training/checkpoints"
    mkdir -p "$DST_DIR/training/summaries"

    # Copy only specific files from checkpoints
    cp "$SRC_DIR/training/checkpoints/model_final.pth" "$DST_DIR/training/checkpoints/"
done

echo "Done."
