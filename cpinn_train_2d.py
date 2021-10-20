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

from modules.pinn_2d import *
from modules.generate_data import *
from modules.utils import *

def train(model_path, figure_path):
    log_path = os.path.join(figure_path, 'log.txt')

    # Points
    points_x = [-1.0, 1.0]
    points_y = [-1.0, 1.0]

    # Set the number of domains
    domain_no = (len(points_x) + len(points_y)) // 2 - 1

    # Set the global left & right boundary of the calculation domain
    global_lb_x = -1.0
    global_rb_x = 1.0
    global_lb_y = -1.0
    global_rb_y = 1.0

    # Initialize CPINN model
    model = CPINN_2D(domain_no, global_lb_x, global_rb_x, global_lb_y, global_rb_y, figure_path)

    # to do
    model.make_domains(points_x, points_y)
    # model.make_boundaries(points)
    # model.plot_domains()
    
    sample = {'Model{}'.format(i+1): PINN(i) for i in range(domain_no)}

    model.module_update(sample)
    

    print(model.domains)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current device:", device)

    b_size = 5
    f_size = 5
    epochs = 10000
    lr = 0.0001
    model.to(device)

    dw = 0.00001
    
    bcs = []
    bcs.append(BCs(b_size, x_lb=-1.0, x_rb=1.0, y_lb=-1.0, y_rb=-1.0, u=0.0, v=0.0, deriv_x=0, deriv_y=0))
    bcs.append(BCs(b_size, x_lb=-1.0, x_rb=1.0, y_lb=1.0, y_rb=1.0, u=0.0, v=0.0, deriv_x=0, deriv_y=0))
    bcs.append(BCs(b_size, x_lb=-1.0, x_rb=-1.0, y_lb=-1.0, y_rb=1.0, u=0.0, v=0.0, deriv_x=0, deriv_y=0))
    bcs.append(BCs(b_size, x_lb=1.0, x_rb=1.0, y_lb=-1.0, y_rb=1.0, u=0.0, v=0.0, deriv_x=0, deriv_y=0))

    pdes = []
    # w1 = lambda, w2: mu
    pdes.append(PDEs(f_size, w1=0, w2=1, fx=0, fy=-1, x_lb=-1.0, x_rb=1.0, y_lb=-1.0, y_rb=1.0))
    
    optims = []
    schedulers = []

    models = model._modules

    for key in models.keys():
        sub_model = models[key]
        optim = torch.optim.Adam(sub_model.parameters(), lr=lr)
        optims.append(optim)
        schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=100, verbose=True))

    dms = model.domains
    
    w_b = 100
    w_f = 1
    w_i = 1

    with open(log_path, 'w') as f:
        f.write("-----------------------------Points-----------------------------\n")
        for p in points_x:
            f.write("x: " + str(p) + "\t")
        for p in points_y:
            f.write("y: " + str(p) + "\t")
        f.write("\n")
        f.write("-----------------------------BCs-----------------------------\n")
        for n, bc in enumerate(bcs):
            f.write("BC {}\n".format(n))
            f.write("size: {}\n".format(bc.size))
            f.write("x: {} ~ {}\n".format(bc.x_lb, bc.x_rb))
            f.write("y: {} ~ {}\n".format(bc.y_lb, bc.y_rb))
            f.write("u: {}\n".format(bc.u))
            f.write("v: {}\n".format(bc.v))
            f.write("deriv_x: {}\n".format(bc.deriv_x))
            f.write("deriv_y: {}\n".format(bc.deriv_y))
        f.write("-----------------------------PDEs-----------------------------\n")
        for n, pde in enumerate(pdes):
            f.write("PDE {}\n".format(n))
            f.write("size: {}\n".format(pde.size))
            f.write("Eq.: {}x(4) - {}\n".format(pde.w1, pde.w2))
            f.write("x: {} ~ {}, y: {} ~ {} \n".format(pde.x_lb, pde.x_rb, pde.y_lb, pde.y_rb))
        f.write("-----------------------------Hyperparameters-----------------------------\n")
        f.write("w_b: {}\n".format(w_b))
        f.write("w_f: {}\n".format(w_f))
        f.write("w_i: {}\n".format(w_i))
        f.write("epochs: {}\n".format(epochs))
        f.write("learning rate: {}\n".format(lr))



    x_bs = []
    y_bs = []
    u_bs = []
    v_bs = []
    x_fs = []
    y_fs = []
    u_fs = []
    v_fs = []

    x_derivs = []
    y_derivs = []
    x_derivs_train = [[] for _ in range(domain_no)]
    y_derivs_train = [[] for _ in range(domain_no)]

    x_bs_train = [[] for _ in range(domain_no)]
    y_bs_train = [[] for _ in range(domain_no)]
    u_bs_train = [[] for _ in range(domain_no)]
    v_bs_train = [[] for _ in range(domain_no)]

    x_fs_train = [[] for _ in range(domain_no)]
    y_fs_train = [[] for _ in range(domain_no)]
    u_fs_train = [[] for _ in range(domain_no)]
    v_fs_train = [[] for _ in range(domain_no)]

    
    pdes_weights = []
    pdes_weights_train = [{} for _ in range(domain_no)]

    for bc in bcs:
        x_b, y_b, u_b, v_b = make_training_boundary_data_2d(size=bc.size, x_lb=bc.x_lb, x_rb=bc.x_rb, y_lb=bc.y_lb, y_rb=bc.y_rb, u=bc.u, v=bc.v)
        x_bs.append(x_b)
        y_bs.append(y_b)
        u_bs.append(u_b)
        v_bs.append(v_b)
        x_derivs.append(torch.ones(x_b.shape).type(torch.IntTensor) * bc.deriv_x)
        y_derivs.append(torch.ones(y_b.shape).type(torch.IntTensor) * bc.deriv_y)

    for pde in pdes:
        x_f, y_f, u_f, v_f = make_training_collocation_data_2d(size=pde.size, x_lb=pde.x_lb, x_rb=pde.x_rb, y_lb=pde.y_lb, y_rb=pde.y_rb)
        x_fs.append(x_f)
        y_fs.append(y_f)
        u_fs.append(u_f)
        v_fs.append(v_f)
        pdes_weights.append((pde.w1, pde.w2, pde.fx, pde.fy))

    for i, dm in enumerate(dms):
        x_lb = dm['x_lb']
        x_rb = dm['x_rb']
        y_lb = dm['y_lb']
        y_rb = dm['y_rb']

        for j, (x_b, y_b) in enumerate(zip(x_bs, y_bs)):
            u_b = u_bs[j]
            x_deriv = x_derivs[j]
            y_deriv = y_derivs[j]
            x_max = bcs[j].x_rb
            y_max = bcs[j].y_rb
            
            if x_lb <= x_max <= x_rb and y_lb <= y_max <= y_rb:
                x_bs_train[i].append(x_b)
                y_bs_train[i].append(y_b)
                u_bs_train[i].append(u_b)
                v_bs_train[i].append(v_b)
                x_derivs_train[i].append(x_deriv)
                y_derivs_train[i].append(y_deriv)
        
        for j, (x_f, y_f) in enumerate(zip(x_fs, y_fs)):
            u_f = u_fs[j]
            x = ( pdes[j].x_lb + pdes[j].x_rb ) / 2
            y = ( pdes[j].y_lb + pdes[j].y_rb ) / 2
            
            pde_weights = pdes_weights[j]
            
            # must be modified when the governing equation is changed
            if x_lb <= x <= x_rb and y_lb <= y <= y_rb:
                x_fs_train[i].append(x_f)
                y_fs_train[i].append(y_f)
                u_fs_train[i].append(u_f)
                v_fs_train[i].append(v_f)
                pdes_weights_train[i]['w1'] = pde_weights[0]
                pdes_weights_train[i]['w2'] = pde_weights[1]
                pdes_weights_train[i]['fx'] = pde_weights[2]
                pdes_weights_train[i]['fy'] = pde_weights[3]
                

    loss_save = np.inf
    
    loss_b_plt = [[] for _ in range(domain_no)]
    loss_f_plt = [[] for _ in range(domain_no)]
    loss_i_plt = [[] for _ in range(domain_no)]
    loss_plt   = [[] for _ in range(domain_no)]

    x_plt = torch.from_numpy(np.arange(global_lb_x, global_rb_x, (global_rb_x - global_lb_x) / 100))
    y_plt = torch.from_numpy(np.arange(global_lb_y, global_rb_y, (global_rb_y - global_lb_y) / 100))

    for epoch in range(epochs):
        for i in range(domain_no):
            optim = optims[i]
            scheduler = schedulers[i]
            optim.zero_grad()

            loss_b = 0.0
            loss_f = 0.0
            loss_i = 0.0
            loss_sum = 0.0
            loss_func = nn.MSELoss()

            x_bs = x_bs_train[i]
            y_bs = y_bs_train[i]
            u_bs = u_bs_train[i]
            v_bs = v_bs_train[i]
            x_derivs = x_derivs_train[i]
            y_derivs = y_derivs_train[i]

            x_fs = x_fs_train[i]
            y_fs = y_fs_train[i]
            u_fs = u_fs_train[i]
            v_fs = v_fs_train[i]
            pde_weights = pdes_weights_train[i]

            for j, (x_b, y_b) in enumerate(zip(x_bs, y_bs)):
                u_b = u_bs[j]
                v_b = v_bs[j]
                x_b = x_b.cuda()
                y_b = y_b.cuda()
                u_b = u_b.cuda()
                v_b = v_b.cuda()
                x_deriv = x_derivs[j]
                y_deriv = y_derivs[j]
                # to be modified when the deriv. is greater than 0
                aa = calc_deriv(x_b, model(x_b, y_b), x_deriv[0])
                loss_b += loss_func(calc_deriv(y_b, aa, y_deriv[0]), torch.cat((u_b, v_b), axis=1)) * w_b

            for j, (x_f, y_f) in enumerate(zip(x_fs, y_fs)):
                u_f = u_fs[j]
                x_f = x_f.cuda()
                y_f = y_f.cuda()
                u_f = u_f.cuda()
                w1 = pde_weights['w1']
                w2 = pde_weights['w2']
                fx = pde_weights['fx']
                fy = pde_weights['fy']
                # print(x_f, u_f, w1, w2)
                u_hat = model(x_f, y_f)[:,0]
                v_hat = model(x_f, y_f)[:,1]
                u_hat_x = calc_deriv(x_f, u_hat, 1)
                u_hat_x_x = calc_deriv(x_f, u_hat_x, 1)
                u_hat_y_y = calc_deriv(y_f, u_hat, 2)
                v_hat_x_x = calc_deriv(x_f, v_hat, 2)
                v_hat_y = calc_deriv(y_f, v_hat, 1)
                v_hat_y_y = calc_deriv(y_f, v_hat_y, 1)

                loss_f = loss_func( (w1 + w2) * calc_deriv(x_f, (u_hat_x + v_hat_y), 1) + w2 * (u_hat_x_x + u_hat_y_y) + fx, u_f)
                loss_f += loss_func( (w1 + w2) * calc_deriv(y_f, (u_hat_x + v_hat_y), 1) + w2 * (v_hat_x_x + v_hat_y_y) + fy, u_f)
                # print("PDEs---------------------")
                # print("w1: {}, w2: {}".format(w1, w2))
                # print(calc_deriv(x_f, model(x_f), 4).item())
                # print(calc_deriv(x_f, model(x_f), 4).item() * w1 - 1÷ * w2, u_f.item())

                # print(x_f, u_f, w1, w2)


            # loss_i = model.get_boundary_error() * w_i
            loss_i = 0.0 

            loss = loss_b + loss_f + loss_i
            loss.backward(retain_graph=True)
            optim.step()
                # print(batch, x_f.shape)
            loss_sum += loss.item()
            loss_b_item = loss_b.item() if torch.is_tensor(loss_b) else loss_b
            loss_b_plt[i].append(loss_b_item)
           
            loss_f_plt[i].append(loss_f.item())
            
            loss_i_item = loss_i.item() if torch.is_tensor(loss_i) else 0.0
            loss_i_plt[i].append(loss_i_item)

            loss_plt[i].append(loss.item())
            # scheduler.step(loss)
            
            if epoch % 50 == 1:
                model.plot_model(x_plt, y_plt)
                # model.plot_separate_models(x_plt, y_plt)

            with torch.no_grad():
                model.eval()
                
                print("Epoch: {0} | LOSS: {1:.5f} | LOSS_B: {2:.5f} | LOSS_F: {3:.5f} | LOSS_I: {4:.5f}".format(epoch+1, loss.item(), loss_b_item, loss_f.item(), loss_i_item))

                if epoch % 50 == 1:
                    model.draw_convergence(epoch + 1, loss_b_plt[i], loss_f_plt[i], loss_i_plt[i], loss_plt[i], i, figure_path)

        if loss_sum < loss_save:
            loss_save = loss_sum
            torch.save(model.state_dict(), model_path)
            print(".......model updated (epoch = ", epoch+1, ")")
        
        # if loss_sum < 0.0000001:
        #     break

                
        
        # print("After 1 epoch {:.3f}s".format(time.time() - start))
            
    print("DONE")

def zip_test():
    x = [1, 2, 3, 4, 5]
    y = [2, 3, 4, 5, 6]

    print(list(enumerate(zip(x, y))))
    for i, (a, b) in enumerate(zip(x, y)):
        print(i, a, b)

def deriv_test():
    x = torch.from_numpy(np.arange(5)).type(torch.FloatTensor).requires_grad_(True)
    y = torch.from_numpy(np.arange(5, 10)).type(torch.FloatTensor).requires_grad_(True)
    z = x + 3 * x * y
    print(calc_deriv(y, calc_deriv(x, z, 1), 1))

def model_test():
    # Points
    points_x = [-1.0, 1.0]
    points_y = [-1.0, 1.0]

    # Set the number of domains
    domain_no = (len(points_x) + len(points_y)) // 2 - 1

    # Set the global left & right boundary of the calculation domain
    global_lb_x = -1.0
    global_rb_x = 1.0
    global_lb_y = -1.0
    global_rb_y = 1.0

    # Initialize CPINN model
    model = CPINN_2D(domain_no, global_lb_x, global_rb_x, global_lb_y, global_rb_y, figure_path=None)

    # to do
    model.make_domains(points_x, points_y)
    # model.make_boundaries(points)
    # model.plot_domains()
    
    sample = {'Model{}'.format(i+1): PINN(i) for i in range(domain_no)}

    model.module_update(sample)

    x = torch.tensor([0, 1, 2]).unsqueeze(0).T.type(torch.FloatTensor)
    y = torch.tensor([1, 2, 3]).unsqueeze(0).T.type(torch.FloatTensor)

    z = model(x, y)
    

def main(model_path, figure_path):
    since = time.time()
    train(model_path, figure_path)
    # deriv_test()
    # model_test()
    # zip_test()
    print("Elapsed time: {:.3f} s".format(time.time() - since))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])