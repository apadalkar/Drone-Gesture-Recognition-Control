import cv2
import time
from gesture_recognition import recognize_gesture  # Placeholder, to be implemented
from debounce import Debouncer  # Placeholder, to be implemented

OUTPUT_HZ = 10

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    debouncer = Debouncer()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            # Placeholder: recognize_gesture should return a command string or None
            gesture = recognize_gesture(frame)
            command = debouncer.update(gesture)
            print(f"Command: {command}")
            # Show the frame (optional, for debugging)
            cv2.imshow('Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(1 / OUTPUT_HZ)
    finally:
        cap.release()
        cv2.destroyAllWindows() 