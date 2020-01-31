import torch
import numpy as np


# Use the GPU if available, or if the memory is insufficient use only the CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

# PyTorch dtype
TORCH_DTYPE = torch.float64

# Numpy dtype
NP_DTYPE = np.float64
