"""
config.py - Configuration file
"""

import torch

### Data path
path = './data'

### Training parameters

learning_rate = 1e-3
num_epochs = 15
batch_size = 128

### Device settings

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = get_device()


### Test functions 

if __name__ == "__main__":
    print (get_device())