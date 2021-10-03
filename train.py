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

    def out_A(self, x):
        out = self.modelA(x) * window(x, 0.4, 0.6, i=0)
        return out
    
    def out_B(self, x):
        out = self.modelB(x) * window(x, 0.4, 0.6, i=1)
        return out
    
    


def train():
    # The number of subdomains
    element_no = 2

    # Elapsed time
    since = time.time()

    # Hyperparameter
    b_size = 100
    f_size = 10000
    epochs = 100000
    w = 20

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current device:", device)

   # Define CombinedPINN
    model = CombinedPINN(2)
    model = nn.DataParallel(model)
    model = model.to(device)
    optims = []
    schedulers = []
    optims.append(torch.optim.Adam(model.module.modelA.parameters(), lr=0.001))
    optims.append(torch.optim.Adam(model.module.modelB.parameters(), lr=0.001))
    schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optims[0], 'min', patience=100, verbose=True))
    schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optims[1], 'min', patience=100, verbose=True))
    x_b, u_b = make_training_boundary_data(b_size // 3, x=0.0, u=0.0)
    x_b_2, u_b_2 = make_training_boundary_data(b_size // 3, x=0.5, u=0.0)
    x_b_3, u_b_3 = make_training_boundary_data(b_size // 3, x=1.0, u=0.0)
    x_f, u_f = make_training_collocation_data(f_size // 2, x_lb=0.0, x_rb=0.6)
    x_f_2, u_f_2 = make_training_collocation_data(f_size // 2, x_lb=0.4, x_rb=1.0)

    x_b = x_b.to(device)
    u_b = u_b.to(device)
    x_b_2 = x_b_2.to(device)
    u_b_2 = u_b_2.to(device)
    x_b_3 = x_b_3.to(device)
    u_b_3 = u_b_3.to(device)
    x_f = x_f.to(device)
    u_f = u_f.to(device)
    x_f_2 = x_f_2.to(device)
    u_f_2 = u_f_2.to(device)

    loss_save = np.inf
    
    loss_b_plt = [[] for _ in range(element_no)]
    loss_f_plt = [[] for _ in range(element_no)]
    loss_plt   = [[] for _ in range(element_no)]


    for epoch in range(epochs):
        for i in range(element_no):
            optim = optims[i]
            scheduler = schedulers[i]
            optim.zero_grad()
            loss_b = torch.zeros(1).to(device)
            loss_f = torch.zeros(1).to(device)
            loss_func = nn.MSELoss()

            loss_sum = torch.zeros(1).to(device)

            if i == 0:
                loss_b += loss_func(model(x_b), u_b) * 1000
                loss_b += loss_func(model(x_b_2), u_b_2) * 1000
                loss_b += loss_func(calc_deriv(x_b, model(x_b), 2), u_b) * 1000
                loss_b += loss_func(calc_deriv(x_b_2, model(x_b_2), 1), u_b_2) * 1000
                loss_f += loss_func(calc_deriv(x_f, model(x_f), 4) - 1, u_f)
            elif i == 1:
                loss_b += loss_func(model(x_b_3), u_b_3) * 1000
                loss_b += loss_func(model(x_b_2), u_b_2) * 1000
                loss_b += loss_func(calc_deriv(x_b_3, model(x_b_3), 2), u_b_3) * 1000
                loss_b += loss_func(calc_deriv(x_b_2, model(x_b_2), 1), u_b_2) * 1000
                loss_f += loss_func(calc_deriv(x_f_2, model(x_f_2), 4) - 1, u_f_2)
            
            loss = loss_f + loss_b
            loss_sum += loss

            loss_b_plt[i].append(loss_b.item())
            loss_f_plt[i].append(loss_f.item())
            loss_plt[i].append(loss.item())

            
            loss.backward()

            optim.step()
            scheduler.step(loss)

            with torch.no_grad():
                model.eval()
            
            print("Epoch: {0} | LOSS: {1:.5f}".format(epoch+1, loss.item()))

            if epoch % 50 == 1:
                draw_convergence(epoch + 1, loss_b_plt[i], loss_f_plt[i], loss_plt[i], i)

        if loss_sum < loss_save:
            loss_save = loss_sum
            torch.save(model.state_dict(), './models/cpinn.data')
            print(".......model updated (epoch = ", epoch+1, ")")
        
        if loss_sum < 0.00001:
            break

        if epoch % 50 == 1:
            draw()
            
    print("Elapsed Time: {} s".format(time.time() - since))
    print("DONE")

def exact(x):
    return torch.sin(20 * x) / 20

def draw_convergence(epoch, loss_b, loss_f, loss, id):
    plt.cla()
    x = np.arange(epoch)

    plt.plot(x, np.array(loss_b), label='Loss_B')
    plt.plot(x, np.array(loss_f), label='Loss_F')
    plt.plot(x, np.array(loss), label='Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig('./figures/convergence_{}.png'.format(id))

def draw():
    model = CombinedPINN(2)
    model = nn.DataParallel(model)
    model.cuda()
    model_path = "./models/cpinn.data"
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    
    x_test_plt = np.arange(10001)/10000
    x_test = torch.from_numpy(x_test_plt).unsqueeze(0).T.type(torch.FloatTensor).cuda()
    
    plt.cla()

    pred = model(x_test).cpu().detach().numpy()

    ex = exact(x_test).cpu().detach().numpy()



    plt.plot(x_test_plt, pred, 'b', label='CPINN')
    # plt.plot(x_test, ex, 'r--', label='Exact')
    plt.legend()
    plt.savefig('./figures/test.png')

    plt.cla()
    plt.plot(x_test_plt, model.module.out_A(x_test).cpu().detach().numpy(), 'b', label='A')
    plt.plot(x_test_plt, model.module.out_B(x_test).cpu().detach().numpy(), 'r--', label='B')
    plt.legend()
    plt.savefig('./figures/separate.png')

    plt.cla()
    plt.plot(x_test_plt, window(x_test, 0.4, 0.6, i=0).cpu().detach().numpy(), 'b', label='A')
    plt.plot(x_test_plt, window(x_test, 0.4, 0.6, i=1).cpu().detach().numpy(), 'r--', label='B')
    plt.legend()
    plt.savefig('./figures/window.png')

def calc_deriv_test():
    x = torch.from_numpy((np.arange(1000)) / 1000).type(torch.FloatTensor)
    x.requires_grad = True
    y = x * x * x * x 
    y_x = calc_deriv(x, window(x, 0.4, 0.6, i=0), 4).detach().numpy()

    plt.cla()
    plt.plot(x.detach().numpy(), y_x, 'b')
    plt.savefig('./figures/deriv_test.png')
    
def main():
    # calc_deriv_test()
    train()
    draw()

if __name__ == '__main__':
    main()








    