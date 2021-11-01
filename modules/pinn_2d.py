import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
import os
import time

from modules.utils import *

class PINN(nn.Module):
    def __init__(self, id):
        super(PINN, self).__init__()

        self.id = id

        self.hidden_layer1      = nn.Linear(2, 40)
        self.hidden_layer2      = nn.Linear(40, 40)
        self.hidden_layer3      = nn.Linear(40, 40)
        self.hidden_layer4      = nn.Linear(40, 40)
        # self.hidden_layer5      = nn.Linear(40, 40)
        self.output_layer       = nn.Linear(40, 2)


    def forward(self, x, y):
        input_data     = torch.cat((x, y), axis=1)
        act_func       = nn.Tanh()
        a_layer1       = act_func(self.hidden_layer1(input_data))
        a_layer2       = act_func(self.hidden_layer2(a_layer1))
        a_layer3       = act_func(self.hidden_layer3(a_layer2))
        a_layer4       = act_func(self.hidden_layer4(a_layer3))
        # a_layer5       = act_func(self.hidden_layer5(a_layer4))
        out            = self.output_layer(a_layer4)

        return out

class BCs():
    def __init__(self, size, x_lb, x_rb, y_lb, y_rb, u, v, deriv_x, deriv_y):
        self.size = size
        self.x_lb = x_lb
        self.x_rb = x_rb
        self.y_lb = y_lb
        self.y_rb = y_rb
        self.u = u
        self.v = v
        self.deriv_x = deriv_x
        self.deriv_y = deriv_y

class PDEs():
    def __init__(self, size, w1, w2, fx, fy, x_lb, x_rb, y_lb, y_rb):
        self.x_lb = x_lb
        self.x_rb = x_rb
        self.y_lb = y_lb
        self.y_rb = y_rb
        self.w1 = w1
        self.w2 = w2
        self.fx = fx
        self.fy = fy
        self.size = size

class Where_2D():
    def __init__(self, dm):
        self.dm = dm
    
    def __call__(self, x, y):
        dm = self.dm
        x_lb = dm['x_lb']
        x_rb = dm['x_rb']
        y_lb = dm['y_lb']
        y_rb = dm['y_rb']
        
        res_x = (x - x_lb) * (x - x_rb)
        res_y = (y - y_lb) * (y - y_rb)

        zeros = torch.zeros(x.shape).cuda()
        ones = torch.ones(x.shape).cuda()
        out_x = torch.where(res_x <= 0, ones, zeros)
        out_y = torch.where(res_y <= 0, ones, zeros)
        return out_x * out_y


class CPINN_2D(nn.Module):
    def __init__(self, domain_no, lb_x, rb_x, lb_y, rb_y, figure_path):
        super(CPINN_2D, self).__init__()
        self.domain_no = domain_no
        self.lb_x = lb_x
        self.rb_x = rb_x
        self.lb_y = lb_y
        self.rb_y = rb_y
        self.figure_path = figure_path
        self.length_x = rb_x - lb_x
        self.length_y = rb_y - lb_y

        self.domains = [{} for _ in range(domain_no)]
        # to do: make boundaries in 2D
        self.boundaries = []
        self.wheres = []
        self.make_wheres()

    def make_wheres(self):
        for i in range(self.domain_no):
            self.wheres.append(Where_2D(self.domains[i]))
    
    def get_wheres(self):
        return self.wheres
    

    def forward(self, x, y):
        out = 0.0
        models = self.get_models()
        if self.domain_no == 1:
            model1 = models["Model1"]
            return model1(x, y)
        # to do: make forward function in 2D
        models_num = []
        where = Where_2D(self.domains)
        for i in range(self.domain_no):
            models_num.append(models["Model{}".format(i+1)])
        for i, model in enumerate(models_num):
            out += model(x, y) * self.wheres[i](x, y)
        
        return out


        # for i in range(self.domain_no - 1):
        #     bd = self.boundaries[i]
        #     where_1 = Where(bd, 1)
        #     where_2 = Where(bd, 0)
            
        #     model1 = models["Model{}".format(i+1)]
        #     model2 = models["Model{}".format(i+2)]
            
        #     out += model1(x) * where_1(x) + model2(x) * where_2(x)
        #     # print("{:.2f}".format(x.item()), where_1(x).item(), where_2(x).item())
        #     # print("{:.2f}".format(out.item()))
        # return out
    
    def module_update(self, dict):
        self.__dict__['_modules'].update(dict)
    
    def get_models(self):
        return self.__dict__['_modules']

    def get_boundary_error_2d(self, size):
        bds = self.boundaries
        if bds == []:
            return 0.0
        out = 0.0
        dw = 0.0001
        for bd in bds:
            x_lb = bd['x_lb']
            x_rb = bd['x_rb']
            y_lb = bd['y_lb']
            y_rb = bd['y_rb']

            if x_lb == x_rb:
                y = torch.from_numpy(np.random.uniform(y_lb, y_rb, size))
                x = torch.ones(y.shape) * x_lb
            elif y_lb == y_rb:
                x = torch.from_numpy(np.random.uniform(x_lb, x_rb, size))
                y = torch.ones(y.shape) * y_lb
            
            x = x.unsqueeze(0).T.type(torch.FloatTensor).cuda().requires_grad_(True)
            y = y.unsqueeze(0).T.type(torch.FloatTensor).cuda().requires_grad_(True)

            if x_lb == x_rb:
                out += ( self(x, y - dw) - self(x, y + dw) ) ** 2
                out += ( calc_deriv(y, self(x, y - dw), 1) - calc_deriv(y, self(x, y + dw), 1) ) ** 2
                # out += ( calc_deriv(y, self(x, y - dw), 2) - calc_deriv(y, self(x, y + dw), 2) ) ** 2
                # out += ( calc_deriv(y, self(x, y - dw), 3) - calc_deriv(y, self(x, y + dw), 3) ) ** 2
            if y_lb == y_rb:
                out += ( self(x - dw, y) - self(x + dw, y) ) ** 2
                out += ( calc_deriv(x, self(x - dw, y), 1) - calc_deriv(x, self(x + dw, y), 1) ) ** 2
                # out += ( calc_deriv(x, self(x - dw, y), 2) - calc_deriv(x, self(x + dw, y), 2) ) ** 2
                # out += ( calc_deriv(x, self(x - dw, y), 3) - calc_deriv(x, self(x + dw, y), 3) ) ** 2
        
        out = torch.sum(out)
        # print(out)

        return out
            


                


    # make domains in 2D
    def make_domains(self, points_x=None, points_y=None):
        domain_no = self.domain_no
        if points_x and points_y:
            for i in range(domain_no):
                self.domains[i]['x_lb'] = points_x[i][0]
                self.domains[i]['x_rb'] = points_x[i][1]
                self.domains[i]['y_lb'] = points_y[i][0]
                self.domains[i]['y_rb'] = points_y[i][1]
                self.domains[i]['id'] = i
                self.domains[i]['adj'] = []

            for i in range(domain_no):
                for j in range(i+1, domain_no):
                    id_1 = self.domains[i]['id']
                    id_2 = self.domains[j]['id']
                    x_lb_1 = self.domains[i]['x_lb']
                    x_rb_1 = self.domains[i]['x_rb']
                    y_lb_1 = self.domains[i]['y_lb']
                    y_rb_1 = self.domains[i]['y_rb']
                    x_lb_2 = self.domains[j]['x_lb']
                    x_rb_2 = self.domains[j]['x_rb']
                    y_lb_2 = self.domains[j]['y_lb']
                    y_rb_2 = self.domains[j]['y_rb']
                    if ( (x_lb_1, x_rb_1) == (x_lb_2, x_rb_2) or (y_lb_1, y_rb_1) == (y_lb_2, y_rb_2) and (x_lb_1, x_rb_1, y_lb_1, y_rb_1) != (x_lb_2, x_rb_2, y_lb_2, y_rb_2)):
                        self.domains[i]['adj'].append(id_2)
                        self.domains[j]['adj'].append(id_1)

    def get_overlapped(self, a, b):
        x_lb_1 = a['x_lb']
        x_rb_1 = a['x_rb']
        y_lb_1 = a['y_lb']
        y_rb_1 = a['y_rb']
        x_lb_2 = b['x_lb']
        x_rb_2 = b['x_rb']
        y_lb_2 = b['y_lb']
        y_rb_2 = b['y_rb']

        if ( (x_lb_1, x_rb_1) == (x_lb_2, x_rb_2) ):
            return x_lb_1, x_rb_1, y_rb_1, y_lb_2
        elif  ( (y_lb_1, y_rb_1) == (y_lb_2, y_rb_2) ):
            return x_rb_1, x_lb_2, y_lb_1, y_rb_1

    # make boundaries in 2D
    def make_boundaries(self):
        domain_no = self.domain_no

        for i in range(domain_no):
            adj = self.domains[i]['adj']
            for a in adj:
                if a > i:
                    x_lb, x_rb, y_lb, y_rb = self.get_overlapped(self.domains[i], self.domains[a])
                    self.boundaries.append({'x_lb': x_lb, 'x_rb': x_rb, 'y_lb': y_lb, 'y_rb': y_rb})
        
        print(self.boundaries)

    # make plotting domains in 2D
    def plot_domains(self):
        dms = self.domains
        bds = self.boundaries

        plt.cla()
        plt.figure(figsize=(6,6))

        for dm in dms:
            x_lb = dm['x_lb']
            x_rb = dm['x_rb']
            y_lb = dm['y_lb']
            y_rb = dm['y_rb']
            x, y = np.meshgrid(np.linspace(x_lb, x_rb, 100), np.linspace(y_lb, y_rb, 100))
            plt.scatter(x, y, label='Subdomain {}'.format(dm['id']))
        
        colors = cm.gray(np.linspace(0, 1, len(bds)))
        for n, bd in enumerate(bds):
            x_lb = bd['x_lb']
            x_rb = bd['x_rb']
            y_lb = bd['y_lb']
            y_rb = bd['y_rb']
            plt.plot((x_lb, x_rb), (y_lb, y_rb), '--', linewidth=4, c=colors[n], label='Interface'.format(n))
        
        plt.legend()
        fpath = os.path.join(self.figure_path, "domains.svg")
        plt.savefig(fpath)

    def plot_separate_models(self, x, y):
        x, y = np.meshgrid(x, y)
        xy = torch.from_numpy(np.vstack((x.flatten(), y.flatten()))).type(torch.FloatTensor)
        plt.cla()
        plt.figure(figsize=(6, 6))
        models = self.get_models()
        for i, key in enumerate(models.keys()):
            model = models[key]
            label = 'Model_{}'.format(i)
            result = model(xy[0].unsqueeze(0).T.cuda(), xy[1].unsqueeze(0).T.cuda()).cpu().detach().numpy()
            plt.scatter(x, y, c=result[:,0], label=label)
        plt.legend()

        fpath = os.path.join(self.figure_path, "separate_models.svg")
        plt.savefig(fpath)

    def plot_model(self, x, y):
        # print(x)
        x, y = np.meshgrid(x, y)
        xy = torch.from_numpy(np.vstack((x.flatten(), y.flatten()))).type(torch.FloatTensor)
        pred = self(xy[0].unsqueeze(0).T.cuda(), xy[1].unsqueeze(0).T.cuda())
        pred_cpu = pred.cpu().detach().numpy()
        plt.cla()
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, c=pred_cpu[:,0])
        cb = plt.colorbar()
        fpath = os.path.join(self.figure_path, "model_x.svg")
        plt.savefig(fpath)
        plt.cla()
        plt.scatter(x, y, c=pred_cpu[:,1])
        fpath = os.path.join(self.figure_path, "model_y.svg")
        plt.savefig(fpath)
        cb.remove()
    
    # to do: make getting boundary error in 2D
    def get_boundary_error(self):
        pass

    def draw_convergence(self, epoch, loss_b, loss_f, loss_i, loss, id, figure_path):
        plt.cla()
        x = np.arange(epoch)

        fpath = os.path.join(figure_path, "convergence_model{}.svg".format(id))

        plt.plot(x, np.array(loss_b), label='Loss_B')
        plt.plot(x, np.array(loss_f), label='Loss_F')
        plt.plot(x, np.array(loss_i), label='Loss_I')
        plt.plot(x, np.array(loss), label='Loss')
        plt.yscale('log')
        plt.legend()
        plt.savefig(fpath)

        csvpath = os.path.join(figure_path, "convergence_model{}.csv".format(id))

        arr = np.array([loss_b, loss_f, loss_i, loss])
        print(arr.shape)
        df = pd.DataFrame(arr.T, columns=['Loss_B', 'Loss_F', 'Loss_I', 'Loss'])
        df.to_csv(csvpath)