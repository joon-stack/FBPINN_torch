import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()

        self.hidden_layer1      = nn.Linear(1, 40)
        self.hidden_layer2      = nn.Linear(40, 40)
        self.hidden_layer3      = nn.Linear(40, 40)
        self.output_layer       = nn.Linear(40, 1)

    def forward(self, x):
        input_data     = x
        act_func       = nn.Sigmoid()
        a_layer1       = act_func(self.hidden_layer1(input_data))
        a_layer2       = act_func(self.hidden_layer2(a_layer1))
        a_layer3       = act_func(self.hidden_layer3(a_layer2))
        out            = self.output_layer(a_layer3)

        return out