import time
from collections import deque, Counter


class Debouncer:
    def __init__(self, window_size=5, min_consistency=0.6, confidence_threshold=0.7):

        self.window_size = window_size
        self.min_consistency = min_consistency
        self.confidence_threshold = confidence_threshold

        self.gesture_history = deque(maxlen=window_size)

        self.current_command = None
        self.command_start_time = 0
        self.last_update_time = 0

        self.activation_threshold = min_consistency
        self.deactivation_threshold = min_consistency * 0.7

        self.min_command_duration = 0.1

    def update(self, gesture, confidence=1.0):
        # update the debouncer with new gesture information.

        current_time = time.time()
        self.last_update_time = current_time

        self.gesture_history.append((gesture, confidence, current_time))

        # clean entries after 1sec
        cutoff_time = current_time - 1.0
        while self.gesture_history and self.gesture_history[0][2] < cutoff_time:
            self.gesture_history.popleft()

        stable_command = self._analyze_gesture_stability()

        return self._update_current_command(stable_command, current_time)

    def _analyze_gesture_stability(self):

        # analyze recent gestures to determine most stable command
        if not self.gesture_history:
            return None

        # confidence threshold filtering
        high_confidence_gestures = [
            (gesture, conf)
            for gesture, conf, _ in self.gesture_history
            if conf >= self.confidence_threshold
        ]

        if not high_confidence_gestures:
            return None

        gesture_weights = {}
        total_weight = 0

        for gesture, confidence in high_confidence_gestures:
            if gesture is not None:
                weight = confidence
                gesture_weights[gesture] = gesture_weights.get(gesture, 0) + weight
                total_weight += weight

        if not gesture_weights or total_weight == 0:
            return None

        # find most weighted gesture
        dominant_gesture = max(gesture_weights.keys(), key=lambda g: gesture_weights[g])
        gesture_ratio = gesture_weights[dominant_gesture] / total_weight

        threshold = (
            self.activation_threshold
            if self.current_command != dominant_gesture
            else self.deactivation_threshold
        )

        return dominant_gesture if gesture_ratio >= threshold else None

    def _update_current_command(self, stable_command, current_time):
        if stable_command != self.current_command:
            # check minimum duration for current command
            if (
                self.current_command is not None
                and current_time - self.command_start_time < self.min_command_duration
            ):
                # keep current command for minimum duration
                return self.current_command

            # transition to new command
            self.current_command = stable_command
            self.command_start_time = current_time

        return self.current_command

    def get_gesture_statistics(self):
        if not self.gesture_history:
            return {"status": "no_data"}

        recent_gestures = [
            g for g, c, t in self.gesture_history if c >= self.confidence_threshold
        ]
        gesture_counts = Counter(recent_gestures)

        stats = {
            "history_length": len(self.gesture_history),
            "high_confidence_count": len(recent_gestures),
            "current_command": self.current_command,
            "gesture_distribution": dict(gesture_counts),
            "time_since_update": time.time() - self.last_update_time,
        }

        if recent_gestures:
            most_common = gesture_counts.most_common(1)[0]
            stats["dominant_gesture"] = most_common[0]
            stats["dominant_ratio"] = most_common[1] / len(recent_gestures)

        return stats

    def reset(self):
        self.gesture_history.clear()
        self.current_command = None
        self.command_start_time = 0
        self.last_update_time = 0


class SimpleDebouncer:
    """
    Simplified debouncer for basic use cases.
    """

    def __init__(self, required_count=3):
        self.required_count = required_count
        self.last_gesture = None
        self.count = 0
        self.current_command = None
