# vision_system.py

import cv2
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation
from ultralytics import YOLO
import config

class VisionSystem:
    def __init__(self):
        print("Initializing Vision System...")
        # Load models
        self.depth_processor = DPTImageProcessor.from_pretrained(config.DEPTH_MODEL_NAME)
        self.depth_model = DPTForDepthEstimation.from_pretrained(config.DEPTH_MODEL_NAME)
        self.yolo_model = YOLO(config.YOLO_MODEL_NAME)
        print("Vision System initialized successfully.")

    def perceive(self, frame):
        """
        Takes a single camera frame and returns a list of detected objects with their data.
        """
        detected_objects = []

        # 1. Depth Estimation
        inputs = self.depth_processor(images=frame, return_tensors="pt")
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        depth_map = prediction.cpu().numpy()

        # 2. Object Detection
        yolo_results = self.yolo_model(frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False)

        # 3. Combine Results
        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                depth_value = depth_map[center_y, center_x]
                
                # We use the inverse of depth; a larger number means closer
                relative_distance = 1 / depth_value if depth_value != 0 else 0

                # Angle is the horizontal deviation from the center
                angle = center_x - (config.FRAME_WIDTH // 2)

                detected_objects.append({
                    "class_name": self.yolo_model.names[int(box.cls[0])],
                    "distance": relative_distance,
                    "angle": angle,
                    "box": (x1, y1, x2, y2)
                })
        
        return detected_objects