import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt

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
        act_func       = nn.Tanh()
        a_layer1       = act_func(self.hidden_layer1(input_data))
        a_layer2       = act_func(self.hidden_layer2(a_layer1))
        a_layer3       = act_func(self.hidden_layer3(a_layer2))
        a_layer4       = act_func(self.hidden_layer3(a_layer3))
        a_layer5       = act_func(self.hidden_layer3(a_layer4))
        out            = self.output_layer(a_layer5)

        # out *= window(input_data, 0.4, 0.6, i=self.id)

        return out
    
def sigmoid(x, a, b, i):
    act = nn.ReLU()
    # def act(x):
    #     return 1 / (1 + torch.exp(-x))
    if i > 0:
        return act((x - a) / (b - a))
    elif i == 0:
        return act((b - x) / (b - a))
    else:
        print("Error")


def relu6(x, a, b, i):
    act_func = nn.ReLU6()
    # act_func = nn.ReLU()
    # return sigmoid((x - a) / 0.01) * sigmoid((b - x) / 0.01)
    # return act_func((x - a) * 6 / (b - a))
    if i > 0:
        return act_func((x - a) * 6 / (b - a)) / 6
        # return act_func((x - a) / 100)
    elif i == 0:
        return act_func((b - x) * 6 / (b - a)) / 6
    else:
        print('Error')

def window():
    pass


def window_test():
    x_test = torch.from_numpy(np.arange(100) / 25)
    pred2 = relu6(x_test, 0.4, 0.6, i=0)
    pred3 = relu6(x_test, 1.4, 1.6, i=0)
    pred4 = relu6(x_test, 2.4, 2.6, i=0)
    # plt.plot(x_test, pred, label='A')
    plt.plot(x_test, pred2 * pred3 * pred4, label='B')
    plt.legend()
    plt.savefig('./figures/window_test.png')
    plt.cla()
