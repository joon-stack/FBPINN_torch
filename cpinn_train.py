import sys
import os

import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import time
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from modules.pinn import *
from modules.generate_data import *
    


def train(model_path, figure_path):
    pass
    

def main(model_path, figure_path):
    since = time.time()
    train(model_path, figure_path)
    print("Elapsed time: {:.3f} s".format(time.time() - since))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])