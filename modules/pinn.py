import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt

class PINN(nn.Module):
    def __init__(self, id):
        super(PINN, self).__init__()

        self.id = id

        self.hidden_layer1      = nn.Linear(1, 40)
        self.hidden_layer2      = nn.Linear(40, 40)
        self.hidden_layer3      = nn.Linear(40, 40)
        self.hidden_layer4      = nn.Linear(40, 40)
        # self.hidden_layer5      = nn.Linear(40, 40)
        self.output_layer       = nn.Linear(40, 1)

    def forward(self, x):
        input_data     = x
        act_func       = nn.Tanh()
        a_layer1       = act_func(self.hidden_layer1(input_data))
        a_layer2       = act_func(self.hidden_layer2(a_layer1))
        a_layer3       = act_func(self.hidden_layer3(a_layer2))
        a_layer4       = act_func(self.hidden_layer4(a_layer3))
        # a_layer5       = act_func(self.hidden_layer5(a_layer4))
        out            = self.output_layer(a_layer4)

        # out *= window(input_data, 0.4, 0.6, i=self.id)

        return out

class BCs():
    def __init__(self, size, x, u, deriv):
        self.size = size
        self.x = x
        self.u = u
        self.deriv = deriv

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

        x = x - (lb + rb) / 2
        res = torch.where(x > 0, zeros, ones) if i > 0 else torch.where(x > 0, ones, zeros)
        out = res.clone().detach().requires_grad_(True).cuda()

        # res = torch.tensor(res).type(torch.FloatTensor).requires_grad_(True).cuda()
        return out
        
class Window():
    def __init__(self):
        self.funcs = []
    
    def __call__(self, x):
        result = torch.ones(x.shape).cuda()
        for func in self.funcs:
            result *= func(x)
        return result / 100
    
    def __str__(self):
        res = ""
        for i, func in enumerate(self.funcs):
            res += ("{} lb: {:.2f}, rb: {:.2f}, i: {} \n".format(i, func.lb, func.rb, func.i))
        return res

    def append_funcs(self, func):
        self.funcs.append(func)
    
class CombinedPINN(nn.Module):
    def __init__(self, domain_no, lb, rb, overlap_size):
        super(CombinedPINN, self).__init__()
        self.domain_no = domain_no
        self.lb = lb
        self.rb = rb
        self.overlap_size = overlap_size
        self.outer_domain_size = 0.0
        self.inner_domain_size = 0.0

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
        plt.savefig('./figures/domain_test.png')

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
                # self.windows[i].append_funcs(Relu(lb, rb, i=idx))
                self.windows[i].append_funcs(Step(lb, rb, i=idx))
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
        plt.savefig("./figures/window_test.png")

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
        plt.savefig('./figures/separate_models.png')

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
        return act((x - a) / (b - a))
    elif i == 0:
        return act((b - x) / (b - a))
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
