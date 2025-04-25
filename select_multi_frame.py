import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

model = torch.hub.load('/home/yangsd/yolov5', 'custom', '/home/yangsd/yolov5s.pt', source='local')
model.conf = 0.25  

def frame_difference(frame1, frame2, size=(64, 64)):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    gray1 = cv2.resize(gray1, size)
    gray2 = cv2.resize(gray2, size)
    
    diff = cv2.absdiff(gray1, gray2)
    return np.sum(diff) / (size[0] * size[1])  # 标准化差异值

def select_special_frames(video_path, diff_threshold=30, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
    
    selected_frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total=frame_count, desc="处理进度") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 每6帧处理1帧（保持原采样逻辑）
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 6 != 0:
                pbar.update(1)
                continue
            
            # YOLO目标检测
            results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(results.xyxy[0]) == 0:  # 没有检测到物体时跳过
                pbar.update(1)
                continue
            
            # 计算与已选帧的最小差异
            min_diff = float('inf')
            for selected in selected_frames:
                current_diff = frame_difference(frame, selected["frame"])
                if current_diff < min_diff:
                    min_diff = current_diff
            
            # 满足阈值条件时保存
            if (len(selected_frames) == 0) or (min_diff > diff_threshold):
                frame_info = {
                    "frame": frame.copy(),  # 存储原始帧的副本
                    "features": cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))
                }
                selected_frames.append(frame_info)
                
                if len(selected_frames) >= max_frames:
                    break
            
            pbar.update(1)
    
    cap.release()
    images = [f["frame"] for f in selected_frames]
    convert_images = []
    for image in images:
        convert_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        convert_images.append(convert_image)

    return convert_images


if __name__ == "__main__":
    video_path = "/home/yangsd/sample.mp4"
    
    selected_frames = select_special_frames(
        video_path=video_path,
        diff_threshold=30,
        max_frames=100
    )
    
    print(f"\n已选取 {len(selected_frames)} 帧")