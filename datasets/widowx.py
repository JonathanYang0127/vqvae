import cv2
import numpy as np
from torch.utils.data import Dataset


class WidowXDataset(Dataset):
    """
    Loads image dataset from ObsDictReplayBuffer
    """
    def __init__(self, replay_buffer, train=True, normalize=False):
        self.data = replay_buffer._obs['image'][:replay_buffer._size].reshape(-1, 3, 48, 48) 
        if normalize:
            self.data /= 255.0
        self.data = self.data.astype(np.float32)

        if train:
            self.data = self.data[:int(0.9 * replay_buffer._size)]
        else:
            self.data = self.data[int(0.1 * replay_buffer._size):]


    def __getitem__(self, index):
        img = self.data[index]
        label = 0
        return img, label

    def __len__(self):
        return len(self.data)

