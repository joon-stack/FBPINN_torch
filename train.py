from numpy.random import f
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import time
import matplotlib.pyplot as plt

from modules.pinn import *
from modules.generate_data import *

class CombinedPINN(nn.Module):
    def __init__(self, element_no):
        super(CombinedPINN, self).__init__()
        self.element_no = element_no
        self.models = []
        
        for _ in range(element_no):
            self.models.append(PINN(id=_))

        self.modelA, self.modelB = self.models
    
    def forward(self, x):
        
        out = self.modelA(x) * window(x, 0.4, 0.6, i=0) + self.modelB(x) * window(x, 0.4, 0.6, i=1)
        return out
    


def train():
    # The number of subdomains
    element_no = 2

    # Elapsed time
    since = time.time()

    # Hyperparameter
    b_size = 100
    f_size = 10000
    epochs = 2000
    w = 20

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current device:", device)

   # Define CombinedPINN
    model = CombinedPINN(2)
    optims = []
    optims.append(torch.optim.Adam(model.modelA.parameters(), lr=0.01))
    optims.append(torch.optim.Adam(model.modelB.parameters(), lr=0.01))

    x_b, u_b = make_training_boundary_data(b_size, x=0.0, u=0.0)
    x_f, u_f = make_training_collocation_data(f_size // 2, x_lb=0.0, x_rb=0.6)
    x_f_2, u_f_2 = make_training_collocation_data(f_size // 2, x_lb=0.4, x_rb=1.0)


    loss_save = np.inf
    
    for epoch in range(epochs):
        for i in range(element_no):
            optim = optims[i]
            optim.zero_grad()
            loss_b = 0.0
            loss_f = 0.0
            loss_func = nn.MSELoss()

            loss_sum = 0.0

            if i == 0:
                loss_b += loss_func(model(x_b), u_b)
                loss_f += loss_func(calc_deriv(x_f, model(x_f), 1) - torch.cos(w * x_f), u_f)
            elif i == 1:
                loss_f += loss_func(calc_deriv(x_f_2, model(x_f_2), 1) - torch.cos(w * x_f_2), u_f_2)
            
            loss = loss_f + loss_b
            loss_sum += loss

            loss.backward()

            optim.step()

            with torch.no_grad():
                model.eval()
            
            print("Epoch: {0} | LOSS: {1:.5f}".format(epoch+1, loss))

        if loss_sum < loss_save:
            loss_save = loss_sum
            torch.save(model.state_dict(), './models/cpinn.data')
            print(".......model updated (epoch = ", epoch+1, ")")
        
        if loss_sum < 0.00001:
            break

        if epoch % 50 == 1:
            draw()

    print("DONE")

def exact(x):
    return torch.sin(20 * x) / 20


def draw():
    model = CombinedPINN(2)
    model_path = "./models/cpinn.data"
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    
    x_test = np.arange(10001)/10000
    x_test = torch.from_numpy(x_test).unsqueeze(0).T.type(torch.FloatTensor)
    
    plt.cla()

    pred = model(x_test).detach().numpy()

    ex = exact(x_test).detach().numpy()

    plt.plot(x_test, pred, 'b', label='CPINN')
    plt.plot(x_test, ex, 'r--', label='Exact')
    plt.legend()
    plt.savefig('./figures/test.png')

    plt.cla()
    modelA
    

    
def main():
    train()
    draw()

if __name__ == '__main__':
    main()








    