import cv2
import base64
import numpy as np
import os
from ultralytics import YOLO
import random


def get_crop(img_path):
    img = cv2.imread(img_path)
    model = YOLO("yolov8s-world.pt")  # or select yolov8m/l-world.pt 
    model.set_classes(["person"])
    results = model.predict(img_path)
    # results[0].show()
    boxes = results[0].boxes.xyxy

    crop_imgs = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        crop_img = img[y1:y2, x1:x2]
        crop_imgs.append(crop_img)
    return crop_imgs


def transform_img(img):
    seed_value = 42
    random.seed(seed_value)
    rotation_prob = 0.5  
    flip_prob = 0.5   

    if random.random() < rotation_prob:
        # 选择一个在 -30 到 30 度之间的随机旋转角度
        angle = random.uniform(-30, 30)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h))
    
    if random.random() < flip_prob:
        # 随机选择翻转类型：1为水平翻转，0为垂直翻转，-1为同时翻转
        flip_mode = random.choice([-1, 0, 1])
        img = cv2.flip(img, flip_mode)   


def main():
    file_path = 'try'
    for root, dir, files in os.walk(file_path):
        idx = 0
        for file in files:
            if file.lower().endswith(('.jpg')):
                img_path = os.path.join(root, file)
                crop_imgs = get_crop(img_path)
                trans_crop_imgs = transform_img(crop_imgs)
            
                for i, img in enumerate(trans_crop_imgs):
                    cls = root.split('\\')[-1]
                    crop_img_path = os.path.join("crop", cls, f"crop_{idx}.jpg")
                    idx += 1
                    os.makedirs(os.path.dirname(crop_img_path), exist_ok=True)
                    cv2.imwrite(crop_img_path, img)
                    print(f"Saved cropped image: {crop_img_path}")

                

if __name__ == "__main__":
    main()

