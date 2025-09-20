# main.py

import cv2
import time
from vision_system import VisionSystem
import config

def main():
    # Initialize our vision system
    vision = VisionSystem()
    
    # Setup the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Variables for FPS calculation
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Get a list of detected objects from the vision system
        detected_objects = vision.perceive(frame)

        # --- FPS Calculation ---
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps_text = f"FPS: {int(fps)}"
        # --- End FPS Calculation ---

        # Print the data to the terminal
        print("--- New Frame ---")
        for obj in detected_objects:
            print(f"Object: {obj['class_name']} | Distance: {obj['distance']:.2f} | Angle: {obj['angle']}px")
        
        # --- Visualization ---
        # Draw boxes and labels for each object
        for obj in detected_objects:
            x1, y1, x2, y2 = obj['box']
            label = f"{obj['class_name']} | {obj['distance']:.1f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the FPS on the frame
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the final frame
        cv2.imshow("Autonomous Car Vision", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()