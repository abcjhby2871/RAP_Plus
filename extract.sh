#!/bin/bash

VIDEO_PATH="messi.mp4"

# CUDA_VISIBLE_DEVICES=0

python my_extract.py --video_path "$VIDEO_PATH"


if [ $? -eq 0 ]; then
    echo "Feature extraction completed successfully."
else
    echo "Feature extraction failed."
fi
