# live_detection.py
import cv2
from ultralytics import YOLO

# --- CALIBRATION CONSTANTS ---
# Replace these with your own values after calibration
KNOWN_OBJECT_WIDTH_CM = 14 # Width of a soda can
FOCAL_LENGTH = 666 # Your calculated focal length from the calibration step

# --- MODEL AND WEBCAM SETUP ---
model = YOLO('yolov8s.pt')
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Get frame dimensions
FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_CENTER_X = FRAME_WIDTH // 2

def estimate_distance(pixel_width):
    """Estimates distance based on the object's width in pixels."""
    if pixel_width == 0:
        return 0
    return (KNOWN_OBJECT_WIDTH_CM * FOCAL_LENGTH) / pixel_width

def calculate_angle_error(box_center_x):
    """Calculates the horizontal error from the frame center."""
    return box_center_x - FRAME_CENTER_X

# --- MAIN LOOP ---
while True:
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, conf=0.5)

    print("--- New Frame ---")
    for r in results:
        for box in r.boxes:
            # Bounding Box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Class and Confidence
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])
            
            # --- CALCULATIONS ---
            pixel_width = x2 - x1
            box_center_x = (x1 + x2) // 2
            
            distance_cm = estimate_distance(pixel_width)
            angle_error_px = calculate_angle_error(box_center_x)

            # Print the complete data
            print(f"Object: {class_name} | Confidence: {confidence:.2f} | "
                  f"Distance: {distance_cm:.2f} cm | Angle Error: {angle_error_px} px")
            
            # --- VISUALIZATION ---
            label = f"{class_name} {confidence:.2f} | {distance_cm:.1f}cm"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("AI Car Vision", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()