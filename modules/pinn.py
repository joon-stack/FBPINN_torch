import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy

from torch.utils.data import Dataset

from modules.utils import *

import os 



class PINN(nn.Module):
    def __init__(self, id):
        super(PINN, self).__init__()

        self.id = id

        self.lb = -1.0
        self.rb = 1.0

        self.hidden_layer1      = nn.Linear(1, 40)
        self.hidden_layer2      = nn.Linear(40, 40)
        self.hidden_layer3      = nn.Linear(40, 40)
        self.hidden_layer4      = nn.Linear(40, 40)
        self.output_layer       = nn.Linear(40, 1)

        # t = ax + b
        self.a = 1
        self.b = 0

    def forward(self, x):
        input_data     = x
        # print("{}x + {}".format(a, b))
        # print("{:.3f}, {:.3f}".format(x.item(), input_data.item()))
        act_func       = nn.Tanh()
        a_layer1       = act_func(self.hidden_layer1(input_data))
        a_layer2       = act_func(self.hidden_layer2(a_layer1))
        a_layer3       = act_func(self.hidden_layer3(a_layer2))
        a_layer4       = act_func(self.hidden_layer4(a_layer3))
        out            = self.output_layer(a_layer4)

        return out

class PINN_surrogate(nn.Module):
    def __init__(self, id):
        super(PINN_surrogate, self).__init__()

        self.id = id

        self.lb = -1.0
        self.rb = 1.0

        self.hidden_layer1      = nn.Linear(2, 40)
        self.hidden_layer2      = nn.Linear(40, 40)
        self.hidden_layer3      = nn.Linear(40, 40)
        self.hidden_layer4      = nn.Linear(40, 40)
        self.output_layer       = nn.Linear(40, 1)

        # t = ax + b
        self.a = 1
        self.b = 0

    def forward(self, x, e):
        input_data     = torch.cat((x, e), axis=1)
        # print("{}x + {}".format(a, b))
        # print("{:.3f}, {:.3f}".format(x.item(), input_data.item()))
        act_func       = nn.Tanh()
        a_layer1       = act_func(self.hidden_layer1(input_data))
        a_layer2       = act_func(self.hidden_layer2(a_layer1))
        a_layer3       = act_func(self.hidden_layer3(a_layer2))
        a_layer4       = act_func(self.hidden_layer4(a_layer3))
        out            = self.output_layer(a_layer4)

        return out

class BCs():
    def __init__(self, size, x, u, deriv, W=None):
        self.size = size
        self.x = x
        self.u = u
        self.deriv = deriv
        self.W = W

class PDEs():
    def __init__(self, size, w1, w2, lb, rb, W=None):
        self.size = size
        self.w1 = w1
        self.w2 = w2
        self.lb = lb
        self.rb = rb
        self.W = W

class Relu():
    def __init__(self, lb, rb, i):
        self.lb = lb
        self.rb = rb
        self.i = i
        self.act_func = nn.ReLU6()
    
    def forward(self, x):
        a = self.lb
        b = self.rb
        i = self.i
        act_func = self.act_func

        if i > 0:
            return act_func((x - a) * 6 / (b - a)) / 6
        elif i == 0:
            return act_func((b - x) * 6 / (b - a)) / 6
        else:
            print('Error')
    
    def __call__(self, x):
        return self.forward(x)
    
class Step(Relu):
    def __init__(self, lb, rb, i):
        super(Step, self).__init__(lb, rb, i)

    def forward(self, x):
        # print(x.shape)
        lb = self.lb
        rb = self.rb
        i = self.i

        zeros = torch.zeros(x.shape).cuda()
        ones  = torch.ones(x.shape).cuda()
        y = x - rb
        x = lb - x
        
        res = torch.where(x > 0, zeros, ones) if i > 0 else torch.where(y < 0, ones, zeros)
        out = res.clone().detach().requires_grad_(True).cuda()

        # res = torch.tensor(res).type(torch.FloatTensor).requires_grad_(True).cuda()
        return out

class Sigmoid(Relu):
    def __init__(self, lb, rb, i):
        self.lb = lb
        self.rb = rb
        self.i = i
        self.act_func = nn.Sigmoid()
    
    def forward(self, x):
        a = self.lb
        b = self.rb
        i = self.i
        act_func = self.act_func

        if i > 0:
            return act_func((x - a) / (b - a) * 10)
        elif i == 0:
            return act_func((b - x) / (b - a) * 10)
        else:
            print('Error')
    
    def __call__(self, x):
        return self.forward(x)
    
class Where():
    def __init__(self, bd, i):
        self.bd = bd
        self.i = i
    
    def __call__(self, x):
        ones = torch.ones(x.shape).cuda()
        zeros = torch.zeros(x.shape).cuda()
        bd = self.bd
        idx = x - bd
        i = self.i
        out = torch.where(idx > 0, zeros, ones) if i > 0 else torch.where(idx > 0, ones, zeros)
        return out

class Window():
    def __init__(self):
        self.funcs = []
    
    def __call__(self, x):
        result = torch.ones(x.shape).cuda()
        for func in self.funcs:
            result *= func(x)
        return result / 1
    
    def __str__(self):
        res = ""
        for i, func in enumerate(self.funcs):
            res += ("{} lb: {:.2f}, rb: {:.2f}, i: {} \n".format(i, func.lb, func.rb, func.i))
        return res

    def append_funcs(self, func):
        self.funcs.append(func)

class CPINN_surrogate(nn.Module):
    def __init__(self, domain_no, lb, rb, figure_path):
        super(CPINN_surrogate, self).__init__()
        self.domain_no = domain_no
        self.lb = lb
        self.rb = rb
        self.figure_path = figure_path
        self.length = rb - lb

        self.domains = [{} for _ in range(self.domain_no)]
        self.boundaries = [ None for _ in range(self.domain_no - 1)]

    def forward(self, x, e):
        out = 0.0

        models = self.get_models()
        if self.domain_no == 1:
            model1 = models["Model1"]
            return model1(x, e)
        
        for i in range(self.domain_no - 1):
            bd = self.boundaries[i]
            where_1 = Where(bd, 1)
            where_2 = Where(bd, 0)
            
            model1 = models["Model{}".format(i+1)]
            model2 = models["Model{}".format(i+2)]
            
            out += model1(x, e) * where_1(x, e) + model2(x, e) * where_2(x, e)
            # print("{:.2f}".format(x.item()), where_1(x).item(), where_2(x).item())
            # print("{:.2f}".format(out.item()))
        return out

    def module_update(self, dict):
        self.__dict__['_modules'].update(dict)

    def get_models(self):
        return self.__dict__['_modules']

    def make_domains(self, points=None):
        if points:
            points_copy = copy.copy(points)

            for i in range(self.domain_no):
                self.domains[i]['lb'] = points_copy[i]
                self.domains[i]['rb'] = points_copy[i + 1]
        else:
            length = self.length
            size = length / self.domain_no
            # print(size)
            for i in range(self.domain_no):
                self.domains[i]['lb'] = self.lb + size * i
                self.domains[i]['rb'] = self.lb + size * (i + 1)

        # to do: make mapping? kind of Jacobian
        length = self.length
        for i in range(self.domain_no):
            # t = ax + b
            lb = self.domains[i]['lb']
            rb = self.domains[i]['rb']
            domain_size = rb - lb
            a = length / domain_size
            self.domains[i]['a'] = a
            self.domains[i]['b'] = -1 - lb * a


    def make_boundaries(self, points=None):
        if points:
            self.boundaries = points[1:-1]
        else:
            lb = self.lb
            length = self.length
            size = length / self.domain_no
            for i in range(self.domain_no - 1):
                self.boundaries[i] = lb + size * (i + 1)
            
        

    def plot_domains(self):
        dms = self.domains
        bds = self.boundaries

        plt.cla()
        fpath = os.path.join(self.figure_path, 'domains.svg')
        for n, dm in enumerate(dms):
            lb = dm['lb']
            rb = dm['rb']
            plt.plot((lb, rb), (0, 0), '--', label='Subdomain {}'.format(n+1))

        # for n, bd in enumerate(bds):
        #     plt.scatter(bd, 0, c=cm.gray(n / len(bds)), label='Boundary {}'.format(n))
        plt.scatter(bds, np.zeros(len(bds)), c='k', label='Interfaces')
        plt.legend()
        plt.savefig(fpath)

    def plot_separate_models(self, x):
        x = x.unsqueeze(0).T.type(torch.FloatTensor).cuda()
        plt.cla()
        models = self.get_models()
        for i, key in enumerate(models.keys()):
            model = models[key]
            label = 'Model_{}'.format(i)
            x_cpu = x.cpu().detach().numpy()
            result = (model(x)).cpu().detach().numpy()
            plt.plot(x_cpu, result, label=label)
        plt.legend()

        fpath = os.path.join(self.figure_path, "separate_models.svg")
        plt.savefig(fpath)
    
    def plot_model(self, x):
        x = x.unsqueeze(0).T.type(torch.FloatTensor).cuda()
        pred = self(x)
        x_cpu = x.cpu().detach().numpy()
        pred_cpu = pred.cpu().detach().numpy()
        plt.cla()
        plt.plot(x_cpu, pred_cpu)
        fpath = os.path.join(self.figure_path, "model.svg")
        plt.savefig(fpath)

    def get_boundary_error(self):
        out = 0.0
        models = self.get_models()
        bds = self.boundaries
        if len(bds) == 0:
            return 0.0
        # print(bds)
        dw = 0.00001
        for i, bd in enumerate(bds, 1):
            
            bd = torch.tensor(bd).unsqueeze(0).T.type(torch.FloatTensor).cuda().requires_grad_(True)
            a = models["Model{}".format(i)](bd - dw)
            b = models["Model{}".format(i + 1)](bd - dw)
            c = models["Model{}".format(i)](bd + dw)
            d = models["Model{}".format(i + 1)](bd + dw)
            out += ( a - b ) ** 2
            out += ( calc_deriv(bd, a, 1) - calc_deriv(bd, b, 1) ) ** 2
            out += ( c - d ) ** 2
            out += ( calc_deriv(bd, c, 1) - calc_deriv(bd, d, 1) ) ** 2
            out += ( calc_deriv(bd, a, 2) - calc_deriv(bd, b, 2) ) ** 2 
            out += ( calc_deriv(bd, c, 2) - calc_deriv(bd, d, 2) ) ** 2 
            out += ( calc_deriv(bd, a, 3) - calc_deriv(bd, b, 3) ) ** 2 
            out += ( calc_deriv(bd, c, 3) - calc_deriv(bd, d, 3) ) ** 2 
  
            P = 0
            # if bd == 0.5:
            #     P = 1
            #     out += ( calc_deriv(bd, a, 3) - calc_deriv(bd, b, 3) + P ) ** 2 
            #     out += ( calc_deriv(bd, c, 3) - calc_deriv(bd, d, 3) + P ) ** 2 
        out /= len(bds)
        return out 

class CPINN(nn.Module):
    def __init__(self, domain_no, lb, rb, figure_path):
        super(CPINN, self).__init__()
        self.domain_no = domain_no
        self.lb = lb
        self.rb = rb
        self.figure_path = figure_path
        self.length = rb - lb

        self.domains = [{} for _ in range(self.domain_no)]
        self.boundaries = [ None for _ in range(self.domain_no - 1)]

    def forward(self, x):
        out = 0.0

        models = self.get_models()
        if self.domain_no == 1:
            model1 = models["Model1"]
            return model1(x)
        
        for i in range(self.domain_no - 1):
            bd = self.boundaries[i]
            where_1 = Where(bd, 1)
            where_2 = Where(bd, 0)
            
            model1 = models["Model{}".format(i+1)]
            model2 = models["Model{}".format(i+2)]
            
            out += model1(x) * where_1(x) + model2(x) * where_2(x)
            # print("{:.2f}".format(x.item()), where_1(x).item(), where_2(x).item())
            # print("{:.2f}".format(out.item()))
        return out

    def module_update(self, dict):
        self.__dict__['_modules'].update(dict)

    def get_models(self):
        return self.__dict__['_modules']

    def make_domains(self, points=None):
        if points:
            points_copy = copy.copy(points)

            for i in range(self.domain_no):
                self.domains[i]['lb'] = points_copy[i]
                self.domains[i]['rb'] = points_copy[i + 1]
        else:
            length = self.length
            size = length / self.domain_no
            # print(size)
            for i in range(self.domain_no):
                self.domains[i]['lb'] = self.lb + size * i
                self.domains[i]['rb'] = self.lb + size * (i + 1)

        # to do: make mapping? kind of Jacobian
        length = self.length
        for i in range(self.domain_no):
            # t = ax + b
            lb = self.domains[i]['lb']
            rb = self.domains[i]['rb']
            domain_size = rb - lb
            a = length / domain_size
            self.domains[i]['a'] = a
            self.domains[i]['b'] = -1 - lb * a


    def make_boundaries(self, points=None):
        if points:
            self.boundaries = points[1:-1]
        else:
            lb = self.lb
            length = self.length
            size = length / self.domain_no
            for i in range(self.domain_no - 1):
                self.boundaries[i] = lb + size * (i + 1)
            
        

    def plot_domains(self):
        dms = self.domains
        bds = self.boundaries

        plt.cla()
        fpath = os.path.join(self.figure_path, 'domains.svg')
        for n, dm in enumerate(dms):
            lb = dm['lb']
            rb = dm['rb']
            plt.plot((lb, rb), (0, 0), '--', label='Subdomain {}'.format(n+1))

        # for n, bd in enumerate(bds):
        #     plt.scatter(bd, 0, c=cm.gray(n / len(bds)), label='Boundary {}'.format(n))
        plt.scatter(bds, np.zeros(len(bds)), c='k', label='Interfaces')
        plt.legend()
        plt.savefig(fpath)

    def plot_separate_models(self, x):
        x = x.unsqueeze(0).T.type(torch.FloatTensor).cuda()
        plt.cla()
        models = self.get_models()
        for i, key in enumerate(models.keys()):
            model = models[key]
            label = 'Model_{}'.format(i)
            x_cpu = x.cpu().detach().numpy()
            result = (model(x)).cpu().detach().numpy()
            plt.plot(x_cpu, result, label=label)
        plt.legend()

        fpath = os.path.join(self.figure_path, "separate_models.svg")
        plt.savefig(fpath)
    
    def plot_model(self, x):
        x = x.unsqueeze(0).T.type(torch.FloatTensor).cuda()
        pred = self(x)
        x_cpu = x.cpu().detach().numpy()
        pred_cpu = pred.cpu().detach().numpy()
        plt.cla()
        plt.plot(x_cpu, pred_cpu)
        fpath = os.path.join(self.figure_path, "model.svg")
        plt.savefig(fpath)

    def get_boundary_error(self):
        out = 0.0
        models = self.get_models()
        bds = self.boundaries
        if len(bds) == 0:
            return 0.0
        # print(bds)
        dw = 0.00001
        for i, bd in enumerate(bds, 1):
            
            bd = torch.tensor(bd).unsqueeze(0).T.type(torch.FloatTensor).cuda().requires_grad_(True)
            a = models["Model{}".format(i)](bd - dw)
            b = models["Model{}".format(i + 1)](bd - dw)
            c = models["Model{}".format(i)](bd + dw)
            d = models["Model{}".format(i + 1)](bd + dw)
            out += ( a - b ) ** 2
            out += ( calc_deriv(bd, a, 1) - calc_deriv(bd, b, 1) ) ** 2
            out += ( c - d ) ** 2
            out += ( calc_deriv(bd, c, 1) - calc_deriv(bd, d, 1) ) ** 2
            out += ( calc_deriv(bd, a, 2) - calc_deriv(bd, b, 2) ) ** 2 
            out += ( calc_deriv(bd, c, 2) - calc_deriv(bd, d, 2) ) ** 2 
            out += ( calc_deriv(bd, a, 3) - calc_deriv(bd, b, 3) ) ** 2 
            out += ( calc_deriv(bd, c, 3) - calc_deriv(bd, d, 3) ) ** 2 
  
            P = 0
            # if bd == 0.5:
            #     P = 1
            #     out += ( calc_deriv(bd, a, 3) - calc_deriv(bd, b, 3) + P ) ** 2 
            #     out += ( calc_deriv(bd, c, 3) - calc_deriv(bd, d, 3) + P ) ** 2 
        out /= len(bds)
        return out 



class CombinedPINN(nn.Module):
    def __init__(self, domain_no, lb, rb, overlap_size, figure_path):
        super(CombinedPINN, self).__init__()
        self.domain_no = domain_no
        self.lb = lb
        self.rb = rb
        self.overlap_size = overlap_size
        self.outer_domain_size = 0.0
        self.inner_domain_size = 0.0
        
        self.figure_path = figure_path

        self.domains = [{} for _ in range(self.domain_no)]
        self.boundaries = [{} for _ in range(self.domain_no - 1)]
        self.windows = [Window() for _ in range(self.domain_no)]

    def module_update(self, dict):
        self.__dict__['_modules'].update(dict)

    def get_models(self):
        return self.__dict__['_modules']

    def make_domains(self):
        length = self.rb - self.lb
        length -= self.overlap_size
        self.outer_domain_size = length / self.domain_no
        self.inner_domain_size = self.outer_domain_size - self.overlap_size
        
        self.domains[0]['lb'] = self.lb
        self.domains[0]['rb'] = self.lb + self.outer_domain_size + self.overlap_size
        self.domains[self.domain_no - 1]['lb'] = self.rb - self.outer_domain_size - self.overlap_size
        self.domains[self.domain_no - 1]['rb'] = self.rb
        for i in range(1, self.domain_no - 1):
            self.domains[i]['lb'] = self.domains[i - 1]['rb'] - self.overlap_size
            self.domains[i]['rb'] = self.domains[i]['lb'] + self.inner_domain_size + self.overlap_size * 2
    
    def make_boundaries(self):
        for i in range(self.domain_no - 1):
            self.boundaries[i]['lb'] = self.domains[i + 1]['lb']
            self.boundaries[i]['rb'] = self.boundaries[i]['lb'] + self.overlap_size

    def plot_domains_and_boundaries(self):
        dms = self.domains
        bds = self.boundaries

        plt.cla()

        for i, dm in enumerate(dms):
            lb = dm['lb']
            rb = dm['rb']
            plt.plot((lb, rb), (0, 0), '--', label='Domain {}'.format(i))
        
        for i, bd in enumerate(bds):
            lb = bd['lb']
            rb = bd['rb']
            plt.plot((lb, rb), (0, 0), label='Boundary {}'.format(i), linewidth=4)
        
        plt.legend()

        fpath = os.path.join(self.figure_path, "domains_and_boundaries.svg")
        plt.savefig(fpath)

    def make_windows(self):
        bds = self.boundaries
        size = self.domain_no
        if size == 1:
            pass
        for i in range(size):
            idx = i
            for j in range(size - 1):
                lb = bds[j]['lb']
                rb = bds[j]['rb']
                self.windows[i].append_funcs(Relu(lb, rb, i=idx))
                # self.windows[i].append_funcs(Step(lb, rb, i=idx))
                # self.windows[i].append_funcs(Sigmoid(lb, rb, i=idx))
                if idx > 0:
                    idx -= 1

    def plot_windows(self):
        plt.cla()
        windows = self.windows
        x_test = torch.from_numpy(np.arange(200) / 100 - 1.0).cuda()
        x_test_plt = x_test.cpu().detach().numpy()
        for i in range(len(windows)):
            window = windows[i]
            
            res = window(x_test).cpu().detach().numpy()
            plt.plot(x_test_plt, res, label=i)
        plt.legend()

        fpath = os.path.join(self.figure_path, "window.svg")
        plt.savefig(fpath)

    def plot_separate_models(self, x):
        x = x.unsqueeze(0).T.type(torch.FloatTensor).cuda()
        plt.cla()
        models = self.get_models()
        for i, key in enumerate(models.keys()):
            model = models[key]
            label = 'Model_{}'.format(i)
            x_cpu = x.cpu().detach().numpy()
            result = (model(x) * self.windows[i](x)).cpu().detach().numpy()
            plt.plot(x_cpu, result, label=label)
        plt.legend()

        fpath = os.path.join(self.figure_path, "separate_models.svg")
        plt.savefig(fpath)
    
    def plot_model(self, x):
        x = x.unsqueeze(0).T.type(torch.FloatTensor).cuda()
        pred = self(x)
        x_cpu = x.cpu().detach().numpy()
        pred_cpu = pred.cpu().detach().numpy()
        plt.cla()
        plt.plot(x_cpu, pred_cpu)
        fpath = os.path.join(self.figure_path, "model.svg")
        plt.savefig(fpath)

    def forward(self, x):
        out = 0.0
        models = self.get_models()
        if self.domain_no == 1:
            model1 = models["Model1"]
            return model1(x)
        
        for i in range(self.domain_no - 1):
            model1 = models["Model{}".format(i+1)]
            model2 = models["Model{}".format(i+2)]
            out += model1(x) * self.windows[i](x) + model2(x) * self.windows[i + 1](x)
        return out

def sigmoid(x, a, b, i):
    act = nn.ReLU()
    # def act(x):
    #     return 1 / (1 + torch.exp(-x))
    if i > 0:
        return act((x - a) / (b - a) / 100)
    elif i == 0:
        return act((b - x) / (b - a) / 100)
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

class BoundaryDataset(Dataset):
    def __init__(self, x, u, d):
        self.x = []
        self.u = []
        self.d = []

        for a in x:
            self.x.extend(a)
        
        for a in u:
            self.u.extend(a)

        for a in d:
            self.d.extend(a)

    
    def __len__(self):
        return len(self.u)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x[idx])
        u = torch.FloatTensor(self.u[idx])
        d = self.d[idx]

        return x, u, d

class PDEDataset(Dataset):
    def __init__(self, x, u):
        self.x = x
        self.u = u
    
    def __len__(self):
        return len(self.u)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x[idx])
        u = torch.FloatTensor(self.u[idx])

        return x, u

def draw_convergence(epoch, loss_b, loss_f, loss, id, figure_path):
    plt.cla()
    x = np.arange(epoch)

    fpath = os.path.join(figure_path, "convergence_model{}.svg".format(id))

    plt.plot(x, np.array(loss_b), label='Loss_B')
    plt.plot(x, np.array(loss_f), label='Loss_F')
    plt.plot(x, np.array(loss), label='Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(fpath)

def draw_convergence_cpinn(epoch, loss_b, loss_f, loss_i, loss, id, figure_path):
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