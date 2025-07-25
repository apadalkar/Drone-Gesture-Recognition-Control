import cv2
import time
from gesture_recognition import recognize_gesture
from gesture_recognition import mp_drawing, mp_hands, hands_detector

OUTPUT_HZ = 10

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            # Mirror the frame 
            frame = cv2.flip(frame, 1)
            # run gesture recognition and get landmarks
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(image_rgb)
            command = None
            if results.multi_hand_landmarks:
                command = recognize_gesture(frame)
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            print("Command: ", command)
            cv2.imshow('Gesture Recognition - Press q to quit', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(1 / OUTPUT_HZ)
    finally:
        cap.release()
        cv2.destroyAllWindows() 