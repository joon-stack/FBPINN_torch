import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

class PINN(nn.Module):
    def __init__(self, id):
        super(PINN, self).__init__()

        self.id = id

        self.hidden_layer1      = nn.Linear(1, 40)
        self.hidden_layer2      = nn.Linear(40, 40)
        self.hidden_layer3      = nn.Linear(40, 40)
        self.hidden_layer4      = nn.Linear(40, 40)
        self.hidden_layer5      = nn.Linear(40, 40)
        self.output_layer       = nn.Linear(40, 1)

    def forward(self, x):
        input_data     = x
        act_func       = nn.Sigmoid()
        a_layer1       = act_func(self.hidden_layer1(input_data))
        a_layer2       = act_func(self.hidden_layer2(a_layer1))
        a_layer3       = act_func(self.hidden_layer3(a_layer2))
        a_layer4       = act_func(self.hidden_layer3(a_layer3))
        a_layer5       = act_func(self.hidden_layer3(a_layer4))
        out            = self.output_layer(a_layer5)

        out *= window(input_data, 0.4, 0.6, i=self.id)

        return out
    
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def window(x, a, b, i):
    act_func = nn.Sigmoid()
    # return sigmoid((x - a) / 0.01) * sigmoid((b - x) / 0.01)
    if i == 1:
        return act_func((x - a) / 0.1)
    elif i == 0:
        return act_func((b - x) / 0.1)

# def window_test():
#     x_test = np.arange(100) / 50
#     pred = window(x_test, 0.4, 0.6)
#     plt.plot(x_test, pred)
#     plt.savefig('./figures/window_test.png')
#     plt.cla()
