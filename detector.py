from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLOWorld
import face_recognition
from PIL import Image
import numpy as np
class Detector_rn50():
    def __init__(self):
        self.processor = DetrImageProcessor.from_pretrained("./detr/facebook/detr-resnet-50", revision="no_timm", cache_dir="./detr/")
        self.model = DetrForObjectDetection.from_pretrained("./detr/facebook/detr-resnet-50", revision="no_timm", cache_dir="./detr/")  
        
    def detect_and_crop(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')

        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

        crops = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(
                    f"Detected {self.model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
            )

            a = image.crop(box)
            crops.append(a)
            a.save(f"logs/crop{box}.jpeg")

        return crops

class Detector():
    def __init__(self):
        self.model = YOLOWorld("yolov8x-worldv2.pt")  # or select yolov8m/l-world.pt for different sizes
        
    # def detect_and_crop(self, image):
    #     # Detect and crop regions of interests
    #     bboxes = self.model.predict(source=image, save=True, conf=0.1)[0].boxes.xyxy
    #     if isinstance(image, str):
    #         image = Image.open(image).convert('RGB')
    #     crops = []
    #     for box in bboxes:
    #         box = [round(i, 2) for i in box.tolist()]
    #         draw = ImageDraw.Draw(image)
    #         draw.rectangle(box, outline="red", width=3)
    #         crop = image.crop(box)
    #         crops.append(crop)
    #     return crops
    
    def detect_and_crop(self, image):
        # Detect and crop regions of interest
        results = self.model.predict(source=image, save=True, conf=0.1)[0]
        bboxes = results.boxes.xyxy  # Bounding boxes (x1, y1, x2, y2)
        class_ids = results.boxes.cls  # Class labels (IDs)
        class_names = results.names  # Class names corresponding to the class IDs

        margin1 = 80
        margin2 = 40

        crops = []
        detected_regions = []  # List to store regions with their class names

        # Add the entire image as a crop
        full_image_bbox = [0, 0, image.width, image.height]  # (x1, y1, x2, y2) for the whole image
        crops.append(image)  # Add the entire image crop
        detected_regions.append({
            "class_name": "full_image",
            "box": full_image_bbox
        })

        if isinstance(class_names, dict):
            class_names = list(class_names.values())

        # Add the full_image information to bboxes, class_ids, and class_names
        full_image_class_id = len(class_names)  # New class ID for "full_image"
        class_ids = torch.cat((class_ids, torch.tensor([full_image_class_id], dtype=torch.long, device=bboxes.device)), dim=0)
        bboxes = torch.cat((bboxes, torch.tensor([full_image_bbox], dtype=torch.float32, device=bboxes.device)), dim=0)
        class_names.append("full_image")

        # Process each bounding box
        for i, (bbox, class_id) in enumerate(zip(bboxes, class_ids)):
            class_name = class_names[int(class_id)]
            x1, y1, x2, y2 = map(int, bbox)
            crop = image.crop((x1, y1, x2, y2))
            crops.append(crop)

            # Store the cropped region and its corresponding class name
            detected_regions.append({
                "class_name": class_name,
                "box": [x1, y1, x2, y2]
            })
            crop_np = np.array(crop)

            # If it's the full image, detect faces in the whole image
            if class_name == "full_image":
                face_locations = face_recognition.face_locations(crop_np)  # Detect faces in full image

                for face_location in face_locations:
                    top, right, bottom, left = face_location

                    # Margin for each detected face
                    top = max(top - margin1, 0)
                    left = max(left - margin2, 0)
                    bottom = min(bottom + margin1, image.height)
                    right = min(right + margin2, image.width)

                    # Crop each face and append it to the crops list
                    face_crop = image.crop((left, top, right, bottom))
                    crops.append(face_crop)

                    # Store the face bounding box and class name for the face
                    detected_regions.append({
                        "class_name": "face",
                        "box": [left, top, right, bottom]
                    })

            # If it's a "person", perform face recognition within the person's crop
            elif class_name == "person":
                face_locations = face_recognition.face_locations(crop_np)  # Detect faces within the crop

                for face_location in face_locations:
                    top, right, bottom, left = face_location

                    # Margin for each detected face
                    top = max(y1 + top - margin1, 0)
                    left = max(x1 + left - margin2, 0)
                    bottom = min(y1 + bottom + margin1, image.height)
                    right = min(x1 + right + margin2, image.width)

                    # Crop each face and append it to the crops list
                    face_crop = image.crop((left, top, right, bottom))
                    crops.append(face_crop)

                    # Store the face bounding box and class name for the face
                    detected_regions.append({
                        "class_name": "face",
                        "box": [left, top, right, bottom]
                    })

        # Draw bounding boxes on the original image and append crops
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        draw = ImageDraw.Draw(image)
        for idx, box in enumerate(bboxes):
            # Get the class label (name) corresponding to the class id
            class_id = int(class_ids[idx])
            class_name = class_names[class_id]

            # Round the bounding box coordinates
            box = [round(i, 2) for i in box.tolist()]

            # Draw the bounding box on the original image
            draw.rectangle(box, outline="red", width=3)

            # Crop the region from the image
            crop = image.crop(box)
            crops.append(crop)

            # Store the cropped region and its corresponding class name
            detected_regions.append({
                "class_name": class_name,
                "box": box
            })

        return crops, detected_regions

if __name__ == "__main__":
    detector = Detector()

    image_path = "hinton.jpg"
    image = Image.open(image_path).convert('RGB')

    crops, detected_regions = detector.detect_and_crop(image)
    for region in detected_regions:
        print(region["class_name"])

    for idx, (crop, detected_region) in enumerate(zip(crops, detected_regions)):
        save_path = f"crop_image/crop_{idx}.jpeg"
        crop.save(save_path)
        print(f"Saved cropped image to {save_path}")