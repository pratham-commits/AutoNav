"""
main_cli.py: CLI-based Autonomous Car Vision System
- Accepts typed instructions
- Shows live video with detection, target selection, and car action
- Uses YOLOv8n and CLIP for robust, general object selection
- Simulates MQTT for car control
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import paho.mqtt.client as mqtt
import threading

# --- CONFIG ---
YOLO_MODEL_PATH = 'yolov8n.pt'  # Use nano for speed
CLIP_MODEL_NAME = 'openai/clip-vit-base-patch16'  # Fast and accurate
MQTT_BROKER = 'localhost'  # Change if needed
MQTT_TOPIC = 'car/control'

# --- MQTT SETUP (Simulated for now) ---
def send_command(command):
    print(f"[MQTT] Command sent: {command}")
    # Uncomment for real MQTT:
    # client = mqtt.Client()
    # client.connect(MQTT_BROKER)
    # client.publish(MQTT_TOPIC, command)
    # client.disconnect()

# --- VISION SYSTEM ---
class VisionSystem:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo = YOLO(YOLO_MODEL_PATH)
        self.clip = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    def detect_objects(self, frame):
        results = self.yolo(frame)[0]
        objects = []
        for box, cls_id, conf in zip(results.boxes.xyxy.cpu().numpy(),
                                     results.boxes.cls.cpu().numpy(),
                                     results.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            class_name = self.yolo.model.names[int(cls_id)]
            objects.append({'box': (x1, y1, x2, y2), 'class_name': class_name, 'conf': float(conf)})
        return objects

    def select_target(self, frame, objects, instruction):
        if not objects:
            return None, None
        crops = [Image.fromarray(frame[y1:y2, x1:x2]) for (x1, y1, x2, y2) in [obj['box'] for obj in objects]]
        inputs = self.processor(text=[instruction]*len(crops), images=crops, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip(**inputs)
            logits_per_image = outputs.logits_per_image.cpu().numpy().flatten()
        best_idx = int(np.argmax(logits_per_image))
        return objects[best_idx], logits_per_image[best_idx]

    def estimate_distance(self, box, frame_shape):
        # Simple: use box height as proxy for distance
        x1, y1, x2, y2 = box
        box_height = y2 - y1
        frame_height = frame_shape[0]
        rel_height = box_height / frame_height
        # Calibrate this formula for your setup
        distance_cm = 100 / (rel_height + 1e-3)
        return distance_cm

# --- MAIN LOOP ---
def main():
    vision = VisionSystem()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    print("System ready. Type your instruction (e.g., 'move to the red book'):")
    instruction = input('> ').strip()
    action = 'waiting'
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error.")
            break
        objects = vision.detect_objects(frame)
        target, score = vision.select_target(frame, objects, instruction)
        # Draw all objects
        for obj in objects:
            x1, y1, x2, y2 = obj['box']
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, obj['class_name'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Highlight target
        if target:
            x1, y1, x2, y2 = target['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"TARGET ({score:.2f})", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            box_height = y2 - y1
            distance = vision.estimate_distance(target['box'], frame.shape)
            cv2.putText(frame, f"Dist: {distance:.1f}cm", (x1, y2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.putText(frame, f"Box Height: {box_height}px", (x1, y2+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            # Simple logic: if target is centered and close, stop; else move/turn
            frame_center = frame.shape[1] // 2
            target_center = (x1 + x2) // 2
            if abs(target_center - frame_center) < 40:
                if distance < 40:
                    action = 'stop'
                else:
                    action = 'move forward'
            elif target_center < frame_center:
                action = 'turn left'
            else:
                action = 'turn right'
            send_command(action)
        else:
            action = 'searching...'
        # Show action on frame
        cv2.putText(frame, f"Action: {action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        cv2.imshow('AutoNav Vision', frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break
        elif key == ord('n'):
            print("Enter new instruction:")
            instruction = input('> ').strip()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
