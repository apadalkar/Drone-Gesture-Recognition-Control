import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [2, 6, 10, 14, 18]

# determine if a finger is extended
def is_finger_extended(landmarks, tip_idx, pip_idx):
    return landmarks[tip_idx].y < landmarks[pip_idx].y

# hand orientation (rough direction of index finger)
def get_hand_direction(landmarks):
    dx = landmarks[8].x - landmarks[0].x
    dy = landmarks[8].y - landmarks[0].y
    
    if abs(dx) > 0.1 and abs(dx) > abs(dy) * 0.5:
        if dx > 0.1:
            return 'right'
        elif dx < -0.1:
            return 'left'
    if dy > 0.07:
        return 'down'
    if dy < -0.1:
        return 'up'
    return None

def recognize_gesture(frame):
    """
    Detects hand landmarks in the given (mirrored) frame using MediaPipe and classifies the gesture.
    Returns a command string (e.g., 'hover', 'move_right', etc.) or None if unclear/no hand.
    """
    # Mirror the frame horizontally for user-friendly interaction
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0].landmark

    # finger states
    fingers = [is_finger_extended(hand_landmarks, tip, pip) for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)]
    thumb, index, middle, ring, pinky = fingers

    # open palm
    if all(fingers):
        return 'hover'
    # pointing gestures (only index extended)
    if index and not middle and not ring and not pinky:
        direction = get_hand_direction(hand_landmarks)
        if direction == 'right':
            return 'move_right'
        elif direction == 'left':
            return 'move_left'
        elif direction == 'up':
            return 'ascend'
        elif direction == 'down':
            return 'descend'
    return None 