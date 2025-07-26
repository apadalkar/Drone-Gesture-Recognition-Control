import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]
FINGER_MIPS = [2, 5, 9, 13, 17]


# determine if a finger is extended
def is_finger_extended(landmarks, finger_idx):
    tip_idx = FINGER_TIPS[finger_idx]
    pip_idx = FINGER_PIPS[finger_idx]
    mcp_idx = FINGER_MIPS[finger_idx]

    if finger_idx == 0:  # thumb edge case (horizontal extension)
        return abs(landmarks[tip_idx].x - landmarks[mcp_idx].x) > 0.04

    else:  # other fingers vertical extension
        tip_above_pip = landmarks[tip_idx].y < landmarks[pip_idx].y - 0.01
        pip_above_mcp = landmarks[pip_idx].y <= landmarks[mcp_idx].y + 0.01
        return tip_above_pip and pip_above_mcp


# hand orientation (rough direction of index finger)
def get_hand_direction(landmarks):
    index_tip = landmarks[8]
    index_mcp = landmarks[5]
    wrist = landmarks[0]

    dx = index_tip.x - index_mcp.x
    dy = index_tip.y - index_mcp.y

    min_threshold = 0.08

    if abs(dx) < min_threshold and abs(dy) < min_threshold:
        return None

    # determine main direction
    if abs(dx) > abs(dy):
        if dx > min_threshold:
            return "right"
        elif dx < -min_threshold:
            return "left"
    else:
        if dy > min_threshold:
            return "down"
        elif dy < -min_threshold:
            return "up"

    return None


def calculate_gesture_confidence(landmarks, gesture_type):
    # Calculate confidence score for detected gesture.

    base_confidence = 0.5

    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    hand_size = abs(wrist.y - middle_mcp.y)

    if hand_size > 0.15:
        base_confidence += 0.2

    if gesture_type in ["move_right", "move_left", "ascend", "descend"]:
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        index_mcp = landmarks[5]

        tip_pip_dist = abs(index_tip.x - index_pip.x) + abs(index_tip.y - index_pip.y)
        pip_mcp_dist = abs(index_pip.x - index_mcp.x) + abs(index_pip.y - index_mcp.y)

        if tip_pip_dist > 0.02 and pip_mcp_dist > 0.02:
            base_confidence += 0.2

    return min(base_confidence, 1.0)


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
        return None, 0.0

    hand_landmarks = results.multi_hand_landmarks[0].landmark

    # finger states
    fingers_extended = [is_finger_extended(hand_landmarks, i) for i in range(5)]
    thumb, index, middle, ring, pinky = fingers_extended

    extended_count = sum(fingers_extended)

    gesture = None
    confidence = 0.0

    # open palm
    if extended_count >= 4:
        gesture = "hover"
        confidence = calculate_gesture_confidence(hand_landmarks, gesture)

    # pointing: index finger extended
    elif index and extended_count <= 2:  # allow for thumb to be extended too
        direction = get_hand_direction(hand_landmarks)
        if direction:
            gesture_map = {
                "right": "move_right",
                "left": "move_left",
                "up": "ascend",
                "down": "descend",
            }
            gesture = gesture_map[direction]
            confidence = calculate_gesture_confidence(hand_landmarks, gesture)

    # return gesture only if confidence is above threshold
    min_confidence_threshold = 0.6
    if confidence >= min_confidence_threshold:
        return gesture, confidence
    else:
        return None, confidence


def get_landmark_drawing_frame(frame):
    # get frame with hand landmarks drawn for visualization
    frame_display = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_display,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
            )

    return frame_display
