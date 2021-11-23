import sys
import os

import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import time
import matplotlib.pyplot as plt


from modules.pinn_2d import *
from modules.generate_data import *
from modules.utils import *

def train(model_path, figure_path):
    log_path = os.path.join(figure_path, 'log.txt')

    # Points
    # points_x = [(-1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, 1.0)]
    # points_y = [(-1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, 1.0)]
    # points_x = [(-1.0, 0.0), (0.0, 1.0)]
    # points_y = [(-1.0, 1.0), (-1.0, 1.0)]
    points_x = [(-1.0, 1.0)]
    points_y = [(-1.0, 1.0)]
    # points_x = [(-1.0, 0.0)]
    # points_y = [(-1.0, 1.0)]

    # Set the number of domains
    domain_no = len(points_x)

    # Set the global left & right boundary of the calculation domain
    global_lb_x = -1.0
    global_rb_x = 1.0
    global_lb_y = -1.0
    global_rb_y = 1.0

    # global_lb_x = 0.0
    # global_rb_x = 1.0
    # global_lb_y = -1.0
    # global_rb_y = 1.0

    # Initialize CPINN model
    model = CPINN_2D_surrogate(domain_no, global_lb_x, global_rb_x, global_lb_y, global_rb_y, figure_path)

    # to do
    model.make_domains(points_x, points_y)
    model.make_boundaries()
    model.plot_domains()
    
    sample = {'Model{}'.format(i+1): PINN_surrogate(i) for i in range(domain_no)}

    model.module_update(sample)
    

    print(model.domains)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current device:", device)

    b_size = 100
    f_size = 10000
    i_size = 100
    epochs = 100000
    lr = 0.0001
    model.to(device)
    
    dw = 0.0

    # E = [2, 4, 6, 8]
    E = [2, 4, 6, 8, 10]

    bcs = []
    pdes = []

    for e in E:
        
        bcs.append(BCs(b_size, x_lb=-1.0, x_rb=0.0, y_lb=-1.0 - dw, y_rb=-1.0 - dw, u=0.0, v=0.0, deriv_x=0, deriv_y=0, E=e))
        bcs.append(BCs(b_size, x_lb=0.0, x_rb=1.0, y_lb=-1.0 - dw, y_rb=-1.0 - dw, u=0.0, v=0.0, deriv_x=0, deriv_y=0, E=e))
        bcs.append(BCs(b_size, x_lb=-1.0, x_rb=0.0, y_lb=1.0 - dw, y_rb=1.0 - dw, u=0.0, v=0.0, deriv_x=0, deriv_y=0, E=e))
        bcs.append(BCs(b_size, x_lb=0.0, x_rb=1.0, y_lb=1.0 - dw, y_rb=1.0 - dw, u=0.0, v=0.0, deriv_x=0, deriv_y=0, E=e))
        bcs.append(BCs(b_size, x_lb=-1.0 - dw, x_rb=-1.0 - dw, y_lb=-1.0, y_rb=1.0, u=0.0, v=0.0, deriv_x=0, deriv_y=0, E=e))
        bcs.append(BCs(b_size, x_lb=1.0 - dw, x_rb=1.0 - dw, y_lb=-1.0, y_rb=1.0, u=0.0, v=0.0, deriv_x=0, deriv_y=0, E=e))
        # bcs.append(BCs(b_size, x_lb=0.0 - dw, x_rb=0.0 - dw, y_lb=-1.0, y_rb=1.0, u=0.0, v=0.0, deriv_x=0, deriv_y=0))

        # bcs.append(BCs(b_size, x_lb=-1.0, x_rb=0.0, y_lb=-1.0 + dw, y_rb=-1.0 + dw, u=0.0, v=0.0, deriv_x=0, deriv_y=0))
        # bcs.append(BCs(b_size, x_lb=0.0, x_rb=1.0, y_lb=-1.0 - dw, y_rb=-1.0 - dw, u=0.0, v=0.0, deriv_x=0, deriv_y=0))
        # bcs.append(BCs(b_size, x_lb=-1.0, x_rb=0.0, y_lb=1.0 + dw, y_rb=1.0 + dw, u=0.0, v=0.0, deriv_x=0, deriv_y=0))
        # bcs.append(BCs(b_size, x_lb=0.0, x_rb=1.0, y_lb=1.0 + dw, y_rb=1.0 + dw, u=0.0, v=0.0, deriv_x=0, deriv_y=0))
        # bcs.append(BCs(b_size, x_lb=-1.0 + dw, x_rb=-1.0 + dw, y_lb=-1.0, y_rb=1.0, u=0.0, v=0.0, deriv_x=0, deriv_y=0))
        # bcs.append(BCs(b_size, x_lb=1.0 + dw, x_rb=1.0 + dw, y_lb=-1.0, y_rb=1.0, u=0.0, v=0.0, deriv_x=0, deriv_y=0))
        # bcs.append(BCs(b_size, x_lb=0.0 + dw, x_rb=0.0 + dw, y_lb=-1.0, y_rb=1.0, u=0.0, v=0.0, deriv_x=0, deriv_y=0))


    # w1 = lambda, w2: mu
        pdes.append(PDEs(f_size, w1=0, w2=e/2, fx=-1, fy=-1, x_lb=-1.0, x_rb=0.0, y_lb=-1.0, y_rb=1.0, E=e))
        pdes.append(PDEs(f_size, w1=0, w2=e/2, fx=-1, fy=-1, x_lb=0.0, x_rb=1.0, y_lb=-1.0, y_rb=1.0, E=e))
    
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
            # f.write("Eq.: {}x(4) - {}\n".format(pde.w1, pde.w2))
            f.write("w1: {}, w2: {}, fx: {}, fy: {}\n".format(pde.w1, pde.w2, pde.fx, pde.fy))
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

    e_bs = []
    e_fs = []

    x_derivs = []
    y_derivs = []
    x_derivs_train = [[] for _ in range(domain_no)]
    y_derivs_train = [[] for _ in range(domain_no)]

    x_bs_train = [[] for _ in range(domain_no)]
    y_bs_train = [[] for _ in range(domain_no)]
    u_bs_train = [[] for _ in range(domain_no)]
    v_bs_train = [[] for _ in range(domain_no)]
    e_bs_train = [[] for _ in range(domain_no)]

    x_fs_train = [[] for _ in range(domain_no)]
    y_fs_train = [[] for _ in range(domain_no)]
    u_fs_train = [[] for _ in range(domain_no)]
    v_fs_train = [[] for _ in range(domain_no)]
    e_fs_train = [[] for _ in range(domain_no)]

    
    pdes_weights = []
    pdes_weights_train = [{} for _ in range(domain_no)]

    for bc in bcs:
        x_b, y_b, u_b, v_b, e_b = make_training_boundary_data_2d_surrogate(size=bc.size, x_lb=bc.x_lb, x_rb=bc.x_rb, y_lb=bc.y_lb, y_rb=bc.y_rb, u=bc.u, v=bc.v, E=bc.E)
        x_bs.append(x_b)
        y_bs.append(y_b)
        # if bc.y_lb == bc.y_rb:
            # v_b = torch.sin(np.pi * x_b).type(torch.FloatTensor)
        u_bs.append(u_b)
        v_bs.append(v_b)
        x_derivs.append(torch.ones(x_b.shape).type(torch.IntTensor) * bc.deriv_x)
        y_derivs.append(torch.ones(y_b.shape).type(torch.IntTensor) * bc.deriv_y)
        e_bs.append(e_b)

    for pde in pdes:
        x_f, y_f, u_f, v_f, e_f = make_training_collocation_data_2d_surrogate(size=pde.size, x_lb=pde.x_lb, x_rb=pde.x_rb, y_lb=pde.y_lb, y_rb=pde.y_rb, E=pde.E)
        x_fs.append(x_f)
        y_fs.append(y_f)
        u_fs.append(u_f)
        v_fs.append(v_f)
        e_fs.append(e_f)
        pdes_weights.append((pde.w1, pde.w2, pde.fx, pde.fy))

    for i, dm in enumerate(dms):
        x_lb = dm['x_lb']
        x_rb = dm['x_rb']
        y_lb = dm['y_lb']
        y_rb = dm['y_rb']

        print(x_lb, x_rb, y_lb, y_rb)
        

        for j, (x_b, y_b) in enumerate(zip(x_bs, y_bs)):
            u_b = u_bs[j]
            v_b = v_bs[j]
            x_deriv = x_derivs[j]
            y_deriv = y_derivs[j]
            x = ( bcs[j].x_lb + bcs[j].x_rb ) / 2
            y = ( bcs[j].y_lb + bcs[j].y_rb ) / 2
            
            
            if x_lb <= x <= x_rb and y_lb <= y <= y_rb:

                x_bs_train[i].append(x_b)
                y_bs_train[i].append(y_b)
                u_bs_train[i].append(u_b)
                v_bs_train[i].append(v_b)
                e_bs_train[i].append(e_b)
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
                e_fs_train[i].append(e_f)
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
            e_bs = e_bs_train[i]
            x_derivs = x_derivs_train[i]
            y_derivs = y_derivs_train[i]

            x_fs = x_fs_train[i]
            y_fs = y_fs_train[i]
            u_fs = u_fs_train[i]
            v_fs = v_fs_train[i]
            e_fs = e_fs_train[i]
            pde_weights = pdes_weights_train[i]

            # print("Domain {}".format(i))
            for j, (x_b, y_b, e_b) in enumerate(zip(x_bs, y_bs, e_bs)):
                u_b = u_bs[j]
                v_b = v_bs[j]
                e_b = e_bs[j]
                x_b = x_b.cuda()
                y_b = y_b.cuda()
                u_b = u_b.cuda()
                v_b = v_b.cuda()
                e_b = e_b.cuda()
                x_deriv = x_derivs[j]
                y_deriv = y_derivs[j]
                # print("Boundary")
                # print("x_max: {:.3f}, x_min: {:.3f}, y_max: {:.3f}, y_min: {:.3f}, u: {:.3f}, v: {:.3f}".format(torch.max(x_b).item(), torch.min(x_b).item(), torch.max(y_b).item(),  torch.min(y_b).item(), torch.max(u_b).item(), torch.max(v_b).item()))
                # to be modified when the deriv. is greater than 0
                aa = calc_deriv(x_b, model(x_b, y_b, e_b), x_deriv[0])

                loss_b += loss_func(calc_deriv(y_b, aa, y_deriv[0]), torch.cat((u_b, v_b), axis=1)) * w_b
            # print(loss_b)

            for j, (x_f, y_f) in enumerate(zip(x_fs, y_fs)):
                u_f = u_fs[j]
                v_f = v_fs[j]
                e_f = e_fs[j]
                x_f = x_f.cuda()
                y_f = y_f.cuda()
                u_f = u_f.cuda()
                v_f = v_f.cuda()
                e_f = e_f.cuda()
                w1 = pde_weights['w1']
                w2 = pde_weights['w2']
                fx = pde_weights['fx']
                fy = pde_weights['fy']
                # print("PDE")
                # print("x_max: {:.3f}, x_min: {:.3f}, y_max: {:.3f}, y_min: {:.3f}, u: {:.3f}, v: {:.3f}".format(torch.max(x_f).item(), torch.min(x_f).item(), torch.max(y_f).item(),  torch.min(y_f).item(), torch.max(u_f).item(), torch.max(v_f).item()))
                # print("x: {:.3f}, y: {:.3f}, u: {:.3f}, v: {:.3f}".format(x_f.item(), y_f.item(), u_f.item(), v_f.item()))
                # print(w1, w2, fx, fy)
                # print(x_f, u_f, w1, w2)
                u_hat = model(x_f, y_f, e_f)[:,0]
                v_hat = model(x_f, y_f, e_f)[:,1]
                u_hat_x = calc_deriv(x_f, u_hat, 1)
                u_hat_x_x = calc_deriv(x_f, u_hat_x, 1)
                u_hat_y_y = calc_deriv(y_f, u_hat, 2)
                v_hat_x_x = calc_deriv(x_f, v_hat, 2)
                v_hat_y = calc_deriv(y_f, v_hat, 1)
                v_hat_y_y = calc_deriv(y_f, v_hat_y, 1)
                loss_f += loss_func( ((w1 + w2) * calc_deriv(x_f, (u_hat_x + v_hat_y), 1) + w2 * (u_hat_x_x + u_hat_y_y) + fx), u_f)
                loss_f += loss_func( ((w1 + w2) * calc_deriv(y_f, (u_hat_x + v_hat_y), 1) + w2 * (v_hat_x_x + v_hat_y_y) + fy), v_f)
                # loss_f = loss_func( ((w1 + w2) * calc_deriv(x_f, (u_hat_x + v_hat_y), 1) + w2 * (u_hat_x_x + u_hat_y_y) + fx * torch.cos(np.pi * y_f / 2) * torch.sin(np.pi * x_f) * 10 * y_f), u_f)
                # loss_f += loss_func( ((w1 + w2) * calc_deriv(y_f, (u_hat_x + v_hat_y), 1) + w2 * (v_hat_x_x + v_hat_y_y) + fy * torch.sin(np.pi * y_f / 2) * torch.cos(np.pi * x_f) * 10 * x_f), v_f)
                loss_f *= w_f
                # print("PDEs---------------------")
                # print("w1: {}, w2: {}".format(w1, w2))
                # print(calc_deriv(x_f, model(x_f), 4).item())
                # print(calc_deriv(x_f, model(x_f), 4).item() * w1 - 1÷ * w2, u_f.item())

                # print(x_f, u_f, w1, w2)


            loss_i = model.get_boundary_error_2d(i_size) * w_i
            # loss_i = 0.0 

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
            
            with torch.no_grad():
                model.eval()
                
                print("Epoch: {0} | LOSS: {1:.5f} | LOSS_B: {2:.5f} | LOSS_F: {3:.5f} | LOSS_I: {4:.5f}".format(epoch+1, loss.item(), loss_b_item, loss_f.item(), loss_i_item))

                if epoch % 50 == 1:
                    model.draw_convergence(epoch + 1, loss_b_plt[i], loss_f_plt[i], loss_i_plt[i], loss_plt[i], i, figure_path)

        if loss_sum < loss_save:
            loss_save = loss_sum
            torch.save(model.state_dict(), model_path)
            print(".......model updated (epoch = ", epoch+1, ")")
            
    print("DONE")

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
    model = CPINN_2D_surrogate(domain_no, global_lb_x, global_rb_x, global_lb_y, global_rb_y, figure_path=None)

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