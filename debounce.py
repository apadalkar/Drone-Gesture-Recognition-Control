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

class SimpleDebouncer:
    """
    Simplified debouncer for basic use cases.
    """
    
    def __init__(self, required_count=3):
        self.required_count = required_count
        self.last_gesture = None
        self.count = 0
        self.current_command = None
