import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands once (singleton pattern)
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def recognize_gesture(frame):
    """
    Detects hand landmarks in the given frame using MediaPipe and classifies the gesture.
    Returns a command string (e.g., 'hover', 'land', 'move_right', etc.) or None if unclear/no hand.
    """
    # Convert the frame to RGB as MediaPipe expects RGB images
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)

    if results.multi_hand_landmarks:
        # TODO: Add gesture classification logic here
        # For now, just return 'hover' if a hand is detected
        return 'hover'
    else:
        return None 