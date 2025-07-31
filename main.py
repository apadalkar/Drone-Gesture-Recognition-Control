import cv2
import time
from gesture_recognition import recognize_gesture, get_landmark_drawing_frame
from debounce import Debouncer

OUTPUT_HZ = 10

def main():

    SHOW_DEBUG_INFO = True
    SHOW_BOUNDING_BOX = True

    cap = cv2.VideoCapture(0)
    debouncer = Debouncer()

    frame_time = 1.0 / OUTPUT_HZ
    last_output_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        current_time = time.time()
        raw_gesture, confidence = recognize_gesture(frame)
        stable_command = debouncer.update(raw_gesture, confidence)

        if current_time - last_output_time >= frame_time:
            print("Command:", stable_command)
            last_output_time = current_time

        display_frame = get_landmark_drawing_frame(frame, show_bounding_box=SHOW_BOUNDING_BOX)

        if SHOW_DEBUG_INFO:
            status_text = f"Command: {stable_command if stable_command else 'None'}"
            cv2.putText(
                display_frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            debug_text = f"Raw: {raw_gesture if raw_gesture else 'None'}"
            cv2.putText(
                display_frame,
                debug_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        cv2.imshow("Gesture Recognition", display_frame)

        # key press handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            SHOW_DEBUG_INFO = not SHOW_DEBUG_INFO
        elif key == ord('b'):
            SHOW_BOUNDING_BOX = not SHOW_BOUNDING_BOX

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

