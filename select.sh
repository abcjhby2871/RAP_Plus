#!/bin/bash

# 设置视频路径、评分文件路径和帧文件路径
VIDEO_PATH="messi.mp4"
SCORE_PATH="output/blip/scores.json"
FRAME_PATH="output/blip/frames.json"
OUTPUT_DIR="images"

# 创建输出目录（如果不存在）
mkdir -p $OUTPUT_DIR

# 执行 Python 脚本
python my_select.py --video_path "$VIDEO_PATH" --score_path "$SCORE_PATH" --frame_path "$FRAME_PATH" --output_file "$OUTPUT_DIR"

# 检查脚本执行是否成功
if [ $? -eq 0 ]; then
    echo "Feature extraction completed successfully."
else
    echo "Feature extraction failed."
fi
