from numpy.lib.npyio import load
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

from modules.pinn import *
from modules.generate_data import *

def exact(x):
    return 1 - x ** 2

def load_training_data(file):
    data = np.loadtxt(fname=file)
    return data

def make_training_data(size):
    # x = torch.randn(size).unsqueeze(0).T
    x = torch.from_numpy( (np.arange(size + 1) * 2 - (size)) / (size)).unsqueeze(0).T.type(torch.FloatTensor)
    # print(x)
    x.requires_grad = True
    u = exact(x)

    return x, u

def calc_loss_2(lambda_1, lambda_2, func, x, y, u_hat):
    u_hat_x = autograd.grad(u_hat.sum(), x, create_graph=True)[0]
    u_hat_xx = autograd.grad(u_hat_x.sum(), x, create_graph=True)[0]

    f = lambda_1 + lambda_2 * u_hat_x + u_hat_xx

    return func(f, y)

def evaluate(epoch):
    model = PINN(id=0)
    model.cuda()
    model_path = "./models/model_infer.data"
    state_dict = torch.load(model_path)
    state_dict = {m.replace('module.', '') : i for m, i in state_dict.items()}
    model.load_state_dict(state_dict)

    x_test_plt = np.arange(101)/100
    x_test = torch.from_numpy(x_test_plt).unsqueeze(0).T.type(torch.FloatTensor).cuda()
    x_test.requires_grad = True
    
    pred = model(x_test).cpu().detach().numpy()
    truth = exact(x_test_plt)

    plt.cla()
    plt.plot(x_test_plt, pred, 'b', label='Inferred')
    # plt.plot(x_test_plt, truth, label='Exact')
    plt.legend()
    plt.savefig('./figures/test_non_pinn.png')
    

    loss = ((pred.T - truth) ** 2).mean()
    print("Epoch {} | Test loss : {:.4f}".format(epoch, loss))

    return loss

def draw_loss(epoch, loss):
    plt.cla()
    plt.plot(loss)
    plt.yscale("log")
    plt.savefig("./figures/test_loss_non_pinn.png")

def train():
    model = PINN(id=0)

    # Prepare to train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current device:", device)

    size = 10
    epochs = 100000

    model.to(device)

    lambda_1 = torch.randn(1).to(device).detach().requires_grad_(True)
    lambda_2 = torch.randn(1).to(device).detach().requires_grad_(True)
    lambda_1 = lambda_1.to(device)
    lambda_2 = lambda_2.to(device)

    # optim = torch.optim.Adam([lambda_1, lambda_2], lr=0.001)
    # optim2 = torch.optim.Adam(model.parameters(), lr=0.001)
    optim = torch.optim.Adam([{'params': model.parameters()},
                            {'params': lambda_1, 'lr': 0.001},
                            {'params': lambda_2, 'lr': 0.001}
                            ],
                            lr=0.001)

    fname = './data/cont_beam.txt'
    data = load_training_data(fname)
    x, _, u = data.T
    x_train = torch.from_numpy(x).unsqueeze(0).T.type(torch.FloatTensor)
    u_train = torch.from_numpy(u).unsqueeze(0).T.type(torch.FloatTensor)
    # x_train, u_train = make_training_data(size)

    x_train = x_train.to(device)
    u_train = u_train.to(device)

    loss_func = nn.MSELoss()

    loss_save = np.inf

    loss_test_plt = []

    for epoch in range(epochs):
        optim.zero_grad()
        # optim2.zero_grad()

        u_hat = model(x_train)
        
        loss_1 = loss_func(u_train, u_hat)
        # loss_2 = calc_loss_2(lambda_1, lambda_2, loss_func, x_train, u_train, u_hat)

        # loss_1.backward()
        # loss_2.backward()

        # loss = loss_1 * 100 + loss_2
        loss = loss_1 * 100
        loss_2 = loss.item()
        loss.backward(retain_graph=True)

        optim.step()
        # optim2.step()

        with torch.no_grad():
            model.eval()

            if loss < loss_save:
                best_epoch = epoch
                loss_save = loss.item()
                torch.save(model.state_dict(), './models/model_infer.data')
                # print(".......model updated (epoch = ", epoch+1, ")")

            if epoch % 10 == 0:    
                print("Epoch: {0} | LOSS: {1:.8f} | LOSS_F: {2:.8f} | Lambda 1: {3:.4f} | Lambda 2 {4:.4f}".format(epoch + 1, loss_1, loss_2, lambda_1.item(), lambda_2.item()))    
            
            if epoch % 100 == 99:
                loss = evaluate(epoch)
                loss_test_plt.append(loss)
                draw_loss(epoch, loss_test_plt)


            # if loss < 1e-6:
            #     break
        
def main():
    train()
    

if __name__ == "__main__":
    main()

