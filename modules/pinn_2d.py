import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
import copy
import os

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
        self.boundaries = None
    
    def forward(self, x, y):
        out = 0.0
        models = self.get_models()
        if self.domain_no == 1:
            model1 = models["Model1"]
            return model1(x, y)
        
        # to do: make forward function in 2D
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

    # to do: make domains in 2D
    def make_domains(self, points_x=None, points_y=None):
        if points_x and points_y:
            self.domains[0]['x_lb']=points_x[0]
            self.domains[0]['x_rb']=points_x[1]
            self.domains[0]['y_lb']=points_y[0]
            self.domains[0]['y_rb']=points_y[1]
        
    # to do: make boundaries in 2D
    def make_boundaries(self, points_x=None, points_y=None):
        if points_x and points_y:
            pass
    
    # to do: make plotting domains in 2D
    def plot_domains(self):
        pass

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

        fpath = os.path.join(self.figure_path, "separate_models.png")
        plt.savefig(fpath)

    def plot_model(self, x, y):
        # print(x)
        x, y = np.meshgrid(x, y)
        xy = torch.from_numpy(np.vstack((x.flatten(), y.flatten()))).type(torch.FloatTensor)
        pred = self(xy[0].unsqueeze(0).T.cuda(), xy[1].unsqueeze(0).T.cuda())
        pred_cpu = pred.cpu().detach().numpy()
        plt.cla()
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, c=pred_cpu[:,1])
        cb = plt.colorbar()
        fpath = os.path.join(self.figure_path, "model.png")
        plt.savefig(fpath)
        cb.remove()
    
    # to do: make getting boundary error in 2D
    def get_boundary_error(self):
        pass

    def draw_convergence(self, epoch, loss_b, loss_f, loss_i, loss, id, figure_path):
        plt.cla()
        x = np.arange(epoch)

        fpath = os.path.join(figure_path, "convergence_model{}.png".format(id))

        plt.plot(x, np.array(loss_b), label='Loss_B')
        plt.plot(x, np.array(loss_f), label='Loss_F')
        plt.plot(x, np.array(loss_i), label='Loss_I')
        plt.plot(x, np.array(loss), label='Loss')
        plt.yscale('log')
        plt.legend()
        plt.savefig(fpath)