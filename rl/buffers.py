import random
import numpy as np


class OffPolicyReplayBuffer:
    def __init__(self, max_len=50000):
        self.max_len = max_len
        self.buffer = []

    def update(self, state, next_state, action, reward, done):
        self.buffer.append([state, next_state, action, reward, done])
        if len(self.buffer) > self.max_len:
            self.buffer.pop(0)

    def sample(self, n=32):
        samples = random.sample(self.buffer, k=min(n, len(self.buffer)))
        samples = [np.vstack([np.expand_dims(sample[i], axis=0) for sample in samples]) for i in range(5)]
        return tuple(samples)