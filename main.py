import cv2
import time
from gesture_recognition import recognize_gesture

OUTPUT_HZ = 10

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            command = recognize_gesture(frame)
            print(f"Command: {command}")
            cv2.imshow('Gesture Recognition - Press q to quit', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(1 / OUTPUT_HZ)
    finally:
        cap.release()
        cv2.destroyAllWindows() 