from collections import deque
from config import WINDOW_SIZE

class SensorBuffer:

    def __init__(self):
        self.buffer = deque(maxlen=WINDOW_SIZE)

    def add(self, data):
        self.buffer.append(data)

    def is_full(self):
        return len(self.buffer) == WINDOW_SIZE

    def get(self):
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()