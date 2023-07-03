from collections import deque
import random
import torch

class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity) # tuple of tuples, the inner tupple represent a single transition

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, done)
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        if batch_size > len(self.buffer):
            return None

        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device) for x in zip(*transitions))
