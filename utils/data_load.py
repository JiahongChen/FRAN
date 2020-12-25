import torch
import torch.utils.data as data

import os
import numpy as np

class MachineSignal(data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = self.data
        label = self.label
        
        return data, label

    def __len__(self):
        return len(self.data)
