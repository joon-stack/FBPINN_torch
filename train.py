import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import time
import matplotlib.pyplot as plt

from modules.pinn import *
from modules.generate_data import *

# def sigmoid(x):
#     return 1 / (1 + torch.exp(-x))


# def window(x, a, b, i):
#     # return sigmoid((x - a) / 0.01) * sigmoid((b - x) / 0.01)
#     if i == 0:
#         return sigmoid((x - a) / 0.01)
#     elif i == 1:
#         return sigmoid((b - x) / 0.01)
    
    


def train():
    # The number of subdomains
    element_no = 2

    # Elapsed time
    since = time.time()

    # Hyperparameter
    b_size = 100
    f_size = 10000
    epochs = 2000

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current device:", device)

    # Models, optimizers, losses, BCs
    models = []
    optims = []
    losses = [0.0] * element_no
    losses_b = [0.0] * element_no
    losses_f = [0.0] * element_no    
    bcs = []
    cols = []

    for i in range(element_no):
        model = PINN(id=i)
        model.to(device)
        models.append(model)
        optims.append(torch.optim.Adam(model.parameters(), lr=0.01))

    # Generate training data: boundary and collocation data
    # x_b, u_b = make_training_boundary_data(b_size, x=0.0, u=0.0)
    # x_b_2, u_b_2 = make_training_boundary_data(b_size, x=0.5, u=0.0)
    # x_b_3, u_b_3 = make_training_boundary_data(b_size, x=1.0, u=0.0)

    # x_f, u_f = make_training_collocation_data(f_size, x_lb=0.0, x_rb=0.6)
    # x_f_2, u_f_2 = make_training_collocation_data(f_size, x_lb=0.4, x_rb=1.0)

    # x_o, u_o = make_training_collocation_data(f_size, x_lb=0.4, x_rb=0.6)
    # # Make all tensors be on the device
    # x_b = x_b.to(device)
    # x_b_2 = x_b_2.to(device)
    # x_b_3 = x_b_3.to(device)
    # u_b = u_b.to(device)
    # u_b_2 = u_b_2.to(device)
    # u_b_3 = u_b_3.to(device)
    # x_f = x_f.to(device)
    # x_f_2 = x_f_2.to(device)
    # u_f = u_f.to(device)
    # u_f_2 = u_f_2.to(device)

    # x_o = x_o.to(device)

    bcs.append(((x_b, u_b, 0), (x_b_2, u_b_2, 0), (x_b, u_b, 2), (x_b_2, u_b_2, 2)))
    bcs.append(((x_b_2, u_b_2, 0), (x_b_3, u_b_3, 0), (x_b_2, u_b_2, 1), (x_b_3, u_b_3, 2)))
    cols.append((x_f, u_f))
    cols.append((x_f_2, u_f_2))



    # Define loss function
    loss_func = nn.MSELoss()

    losses_save = [np.inf] * element_no

    model = models[0]

    # Train NNs
    # for epoch in range(epochs):
    #     optim = optims[0]
    #     boundary = bcs[0]
    #     loss_b = losses_b[0]
    #     loss_f = losses_f[0]
    #     loss = losses[0]
    #     loss_save = losses_save[0]

        # optim.zero_grad()
        # for bc in boundary:
        #     x_b_train, u_b_train, times = bc
        #     loss_b += loss_func(calc_deriv(x_b_train, model(x_b_train), times), u_b_train)
        # x_f_train, u_f_train = cols[0]
        # loss_f += loss_func(calc_deriv(x_f_train, model(x_f_train), 4) - 1, u_f_train)

        # loss = loss_b + loss_f
        # loss.backward()
        # optim.step()

        # with torch.no_grad():
        #     model.eval()

        #     if loss < loss_save:
        #         loss_save = loss
        #         torch.save(model.state_dict(), './models/model_{}.data'.format(0))
        #         print(".......model {} updated (epoch = ".format(0), epoch+1, ")")
        #     print("Model {0} | Epoch: {1} | Loss: {2}".format(0, epoch+1, loss))

        #     if loss < 0.00001:
        #         break

    loss_b_plt = []
    loss_f_plt = []
    loss_plt = []



    for epoch in range(epochs):
        for optim in optims:
            optim.zero_grad()

        for i, model in enumerate(models):
            model.train()
            optim = optims[i]
            boundary = bcs[i]
            loss_b = losses_b[i]
            loss_f = losses_f[i]
            loss = losses[i]
            loss_save = losses_save[i]
            optim.zero_grad()
            
            

            for bc in boundary:
                x_b_train, u_b_train, times = bc
                model_b = model(x_b_train)
                loss_b += loss_func(calc_deriv(x_b_train, model_b, times), u_b_train)
            x_f_train, u_f_train = cols[0]
            model_f = model(x_f_train)
            loss_f += loss_func(calc_deriv(x_f_train, model_f, 4) - 1, u_f_train)


            loss = loss_b + loss_f
            loss.backward()
            optim.step()

            with torch.no_grad():
                model.eval()

                if loss < loss_save:
                    losses_save[i] = loss
                    torch.save(model.state_dict(), './models/model_{}.data'.format(i))
                    print(".......model {} updated (epoch = ".format(i), epoch+1, ")")
                print("Model {0} | Epoch: {1} | Loss: {2}".format(i, epoch+1, loss))
            
            if epoch % 100 == 1:
                draw()

                # if loss < 0.00001:
                #     continue
    print("Elapsed time: {:.3f} s".format(time.time() - since))
    print("Done!") 

def draw():
    plt.cla()
    element_no = 2
    for i in range(element_no):
        x_test = np.arange(10001)/10000
        x_test = torch.from_numpy(x_test).unsqueeze(0).T.type(torch.FloatTensor)
        pred = window(x_test, 0.4, 0.6, i=i).detach().numpy()
        plt.plot(x_test, pred, label="Window_{}".format(i))
    plt.legend()
    plt.savefig('./figures/window.png')
    plt.cla()

    
    for i in range(element_no):
        model = PINN(id=i)
        model_path = "./models/model_{}.data".format(i)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        x_test = np.arange(10001)/10000
        x_test_tensor = torch.from_numpy(x_test).unsqueeze(0).T.type(torch.FloatTensor)
        pred = model(x_test_tensor).detach().numpy()
        plt.plot(x_test, pred, label="NN_{}".format(i))
    plt.legend()
    plt.savefig('./figures/fig.png')

    
def main():
    train()
    draw()

if __name__ == '__main__':
    main()








    