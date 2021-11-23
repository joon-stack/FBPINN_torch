import sys
import os

import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import time
import matplotlib.pyplot as plt



from modules.pinn import *
from modules.generate_data import *
from modules.utils import *
    

def train(model_path, figure_path):
    log_path = os.path.join(figure_path, 'log.txt')

    # Points
    # points = [-1.0, 0.0 - dx, 0.0 + dx, 1.0]
    points = [-1.0,1.0]
    # points = [-1.0, -0.5, 0.0, 0.5, 1.0]
    # points = [-0.5, 0.5]
    # points = [0.5, 1.0]
    # Set the number of domains
    domain_no = len(points) - 1

    # Set the global left & right boundary of the calculation domain
    global_lb = -1.0
    global_rb = 1.0

    # Initialize CPINN model
    model = CPINN_surrogate(domain_no, global_lb, global_rb, figure_path)
    model.make_domains(points)
    model.make_boundaries(points)
    model.plot_domains()

    dms = model.domains
    
    sample = {'Model{}'.format(i+1): PINN_surrogate(i) for i in range(domain_no)}

    # for i, dm in enumerate(dms):
    #     a = dm['a']
    #     b = dm['b']
    #     sample['Model{}'.format(i+1)].a = 1
    #     sample['Model{}'.format(i+1)].b = 0

    model.module_update(sample)
    

    # print(model.domains)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current device:", device)

    b_size = 100
    f_size = 100
    epochs = 10000
    lr = 0.0001
    model.to(device)

    dw = 0.0
    W = [0.1, 0.5, 1, 2, 3, 4, 5, 10]
    # W = [0.1, 0.5, 1]
    bcs = []
    pdes = []

    for w in W:
    
        bcs.append(BCs(b_size, x=-1.0 + dw, u=0.0, deriv=0, W=w))
        # bcs.append(BCs(b_size, x=1.0 + dw, u=0.0, deriv=0))
        bcs.append(BCs(b_size, x=-1.0 + dw, u=0.0, deriv=2, W=w))
        bcs.append(BCs(b_size, x=1.0 + dw, u=0.0, deriv=0, W=w))
        bcs.append(BCs(b_size, x=1.0 + dw, u=0.0, deriv=2, W=w))

        # bcs.append(BCs(b_size, x=-1.0 - dw, u=0.0, deriv=0, W=w))
        # # bcs.append(BCs(b_size, x=1.0 - dw, u=0.0, deriv=0))
        # bcs.append(BCs(b_size, x=-1.0 - dw, u=0.0, deriv=2, W=w))
        # bcs.append(BCs(b_size, x=1.0 - dw, u=0.0, deriv=0, W=w))
        # bcs.append(BCs(b_size, x=1.0 - dw, u=0.0, deriv=2, W=w))

      
        pdes.append(PDEs(f_size, w1=1, w2=w, lb=-1.0, rb=1.0, W=w))

    
    optims = []
    schedulers = []

    models = model._modules

    for key in models.keys():
        sub_model = models[key]
        optim = torch.optim.Adam(sub_model.parameters(), lr=lr)
        optims.append(optim)
        schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=100, verbose=True))

    dms = model.domains
    
    w_b = 1
    w_f = 1
    w_i = 1

    with open(log_path, 'w') as f:
        f.write("-----------------------------Points-----------------------------\n")
        for p in points:
            f.write(str(p) + "\t")
        f.write("\n")
        f.write("-----------------------------BCs-----------------------------\n")
        for n, bc in enumerate(bcs):
            f.write("BC {}\n".format(n))
            f.write("size: {}\n".format(bc.size))
            f.write("x: {}\n".format(bc.x))
            f.write("u: {}\n".format(bc.u))
            f.write("deriv: {}\n".format(bc.deriv))
        f.write("-----------------------------PDEs-----------------------------\n")
        for n, pde in enumerate(pdes):
            f.write("PDE {}\n".format(n))
            f.write("size: {}\n".format(pde.size))
            f.write("Eq.: {}x(4) - {}\n".format(pde.w1, pde.w2))
            f.write("Boundary: {} to {}\n".format(pde.lb, pde.rb))
        f.write("-----------------------------Hyperparameters-----------------------------\n")
        f.write("w_b: {}\n".format(w_b))
        f.write("w_f: {}\n".format(w_f))
        f.write("w_i: {}\n".format(w_i))
        f.write("epochs: {}\n".format(epochs))
        f.write("learning rate: {}\n".format(lr))



    x_bs = []
    u_bs = []
    x_fs = []
    u_fs = []
    e_bs = []
    e_fs = []

    x_derivs = []
    x_derivs_train = [[] for _ in range(domain_no)]

    x_bs_train = [[] for _ in range(domain_no)]
    u_bs_train = [[] for _ in range(domain_no)]
    e_bs_train = [[] for _ in range(domain_no)]

    x_fs_train = [[] for _ in range(domain_no)]
    u_fs_train = [[] for _ in range(domain_no)]
    e_fs_train = [[] for _ in range(domain_no)]

    
    pdes_weights = []
    # pdes_weights_train = [{} for _ in range(domain_no)]
    pdes_weights_train = [[] for _ in range(domain_no)]

    for bc in bcs:
        x_b, u_b, e_b = make_training_boundary_data_surrogate(b_size=bc.size, x=bc.x, u=bc.u, w=bc.W)
        x_bs.append(x_b)
        u_bs.append(u_b)
        e_bs.append(e_b)
        x_derivs.append(torch.ones(x_b.shape).type(torch.IntTensor) * bc.deriv)

    for pde in pdes:
        x_f, u_f, e_f = make_training_collocation_data_surrogate(f_size=pde.size, x_lb=pde.lb, x_rb=pde.rb, w=pde.W)
        x_fs.append(x_f)
        u_fs.append(u_f)
        e_fs.append(e_f)
        pdes_weights.append((pde.w1, pde.w2))
        # pdes_weights.append((1, 1))

    

    for i, dm in enumerate(dms):
        lb = dm['lb']
        rb = dm['rb']
        # t = ax + b
        a = dm['a']
        b = dm['b']
        # print(a, b)

        for j, x_b in enumerate(x_bs):
            u_b = u_bs[j]
            e_b = e_bs[j]
            x_deriv = x_derivs[j]
            x = x_b[0]
            # print(lb, rb)
            if lb <= x <= rb:
                x_bs_train[i].append(x_b)
                u_bs_train[i].append(u_b)
                e_bs_train[i].append(e_b)
                x_derivs_train[i].append(x_deriv)
        
        for j, x_f in enumerate(x_fs):
            u_f = u_fs[j]
            e_f = e_fs[j]
            x = ( pdes[j].lb + pdes[j].rb ) / 2
            
            pde_weights = pdes_weights[j]
            # print(pde_weights)
            # must be modified when the governing equation is changed
            if lb <= x <= rb:
                # print(lb, x, rb, i)
                x_fs_train[i].append(x_f)
                u_fs_train[i].append(u_f)
                e_fs_train[i].append(e_f)
                pdes_weights_train[i].append(pde_weights)
                # pdes_weights_train[i]['w1'] = pde_weights[0]
                # pdes_weights_train[i]['w2'] = pde_weights[1]

    # print(pdes_weights_train)
    # print(x_bs_train)
    # print(x_fs_train)
    loss_save = np.inf
    
    loss_b_plt = [[] for _ in range(domain_no)]
    loss_f_plt = [[] for _ in range(domain_no)]
    loss_i_plt = [[] for _ in range(domain_no)]
    loss_plt   = [[] for _ in range(domain_no)]

    x_plt = torch.from_numpy(np.arange((global_rb - global_lb) * 1000) / 1000 + global_lb) 

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
            u_bs = u_bs_train[i]
            e_bs = e_bs_train[i]
            x_derivs = x_derivs_train[i]

            x_fs = x_fs_train[i]
            u_fs = u_fs_train[i]
            e_fs = e_fs_train[i]
            pde_weights = pdes_weights_train[i]

            for j, x_b in enumerate(x_bs):
                u_b = u_bs[j]
                e_b = e_bs[j]
                x_b = x_b.cuda()
                u_b = u_b.cuda()
                e_b = e_b.cuda()
                x_deriv = x_derivs[j]
                # print(x_deriv)
                # print(x_b, u_b, x_deriv)
                # print(model(x_b).item(), u_b.item())
                loss_b += loss_func(calc_deriv(x_b, model(x_b, e_b), x_deriv[0]), u_b) * w_b
                # print("BCs---------------------")
                # print(calc_deriv(x_b, model(x_b), x_deriv[0]).item())
                # print(model(x_b).item(), u_b.item())
                # print(loss_b.item())
            
            for j, x_f in enumerate(x_fs):
                # print(j)
                u_f = u_fs[j]
                e_f = e_fs[j]
                x_f = x_f.cuda()
                u_f = u_f.cuda()
                e_f = e_f.cuda()
                # w1 = pde_weights['w1']
                # w2 = pde_weights['w2']
                w1 = pde_weights[j][0]
                w2 = pde_weights[j][1]
                # print(x_f, u_f, w1, w2)
                zeros = torch.zeros(x_f.shape).cuda()
                ones = torch.ones(x_f.shape).cuda()
                # loss_f += loss_func(calc_deriv(x_f, model(x_f), 4) * w1 - 1 * w2 * ( torch.where(0 > x_f, ones * torch.sin(-np.pi * x_f), zeros) + torch.where(0.25 < x_f, ones, zeros) * torch.where(0.75 > x_f, ones, zeros)), u_f) * w_f
                # loss_f += loss_func(calc_deriv(x_f, model(x_f), 4) * w1 - 1 * w2 * torch.cos(x_f * np.pi / 2), u_f) * w_f
                loss_f += loss_func(calc_deriv(x_f, model(x_f, e_f), 4) * w1 - 1 * w2, u_f) * w_f

                # print(x_f.item(), u_f.item(), e_f.item())
                
                # print("PDEs---------------------")
                # print("w1: {}, w2: {}".format(w1, w2))
                # print(calc_deriv(x_f, model(x_f), 4).item())
                # print(calc_deriv(x_f, model(x_f), 4).item() * w1 - 1÷ * w2, u_f.item())

                # print(x_f, u_f, w1, w2)


            loss_i = model.get_boundary_error() * w_i
            # loss_i = 0.0 

            loss = loss_b + loss_f + loss_i
            loss.backward()
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
            
            # if epoch % 50 == 1:
            #     model.plot_model(x_plt)
            #     model.plot_separate_models(x_plt)

            with torch.no_grad():
                model.eval()
                
                print("Epoch: {0} | LOSS: {1:.5f} | LOSS_B: {2:.5f} | LOSS_F: {3:.5f} | LOSS_I: {4:.5f}".format(epoch+1, loss.item(), loss_b_item, loss_f.item(), loss_i_item))

                if epoch % 50 == 1:
                    draw_convergence_cpinn(epoch + 1, loss_b_plt[i], loss_f_plt[i], loss_i_plt[i], loss_plt[i], i, figure_path)

        if loss_sum < loss_save:
            loss_save = loss_sum
            torch.save(model.state_dict(), model_path)
            print(".......model updated (epoch = ", epoch+1, ")")
        
        # if loss_sum < 0.0000001:
        #     break

                
        
        # print("After 1 epoch {:.3f}s".format(time.time() - start))
            
    print("DONE")

    

def main(model_path, figure_path):
    since = time.time()
    train(model_path, figure_path)
    print("Elapsed time: {:.3f} s".format(time.time() - since))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])