from numpy.random import f
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import time
import matplotlib.pyplot as plt

from modules.pinn import *
from modules.generate_data import *

class BCs():
    def __init__(self, size, x, u, deriv):
        self.size = size
        self.x = x
        self.u = u
        self.deriv = deriv

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
        self.windows = [torch.zeros(1) for _ in range(self.domain_no)]

        self.window_made = False
    
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

    def make_windows(self, x):
        bds = self.boundaries
        dms = self.domains
        size = len(dms)
        for i in range(size):
            idx = i
            for j in range(size - 1):
                lb = bds[j]['lb']
                rb = bds[j]['rb']
                if j == 0:
                    self.windows[i] = relu6(x, lb, rb, i=idx)
                else:
                    self.windows[i] *= relu6(x, lb, rb, i=idx)
                if idx > 0:
                    idx -= 1
            self.windows[i] /= 100
        
        self.window_made = True

    def plot_windows(self):
        plt.cla()
        windows = self.windows
        x_test = torch.from_numpy(np.arange(100) / 100)
        for i in range(len(windows)):
            plt.plot(x_test, windows[i], label=i)
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
            self.make_windows(x)
            result = (model(x) * self.windows[i]).cpu().detach().numpy()
            plt.plot(x_cpu, result, label=label)
        plt.legend()
        plt.savefig('./figures/separate_models.png')

    def forward(self, x):
        out = 0.0
        models = self.get_models()
        self.make_windows(x)
        for i in range(self.domain_no - 1):
            lb = self.boundaries[i]['lb']
            rb = self.boundaries[i]['rb']
            model1 = models["Model{}".format(i+1)]
            model2 = models["Model{}".format(i+2)]
            # print(x)
            # print(model1(x))
            # print(self.windows[i])
            out += model1(x) * self.windows[i] + model2(x) * self.windows[i + 1]
        return out


def train():
    # Set the starting time
    since = time.time()

    # Set the number of domains
    domain_no = 2

    # Set the global left & right boundary of the calculation domain
    global_lb = 0.0
    global_rb = 1.0

    # Set the size of the overlapping area between domains
    overlap_size = 0.2

    # Initialize combined PINNs
    test = CombinedPINN(domain_no, global_lb, global_rb, overlap_size)
    sample = {'Model{}'.format(i+1): PINN(i) for i in range(domain_no)}
    test.module_update(sample)
    test.make_domains()
    test.make_boundaries()
    test.plot_domains_and_boundaries()

    # Test windows
    x_test = torch.from_numpy(np.arange(100) / 100)
    test.make_windows(x_test)
    test.plot_windows()

    
    # Prepare to train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current device:", device)
    b_size = 100
    f_size = 10000
    epochs = 10000
    # test = nn.DataParallel(test)
    test.to(device)

    # Set boundary conditions
    bcs = []
    bcs.append(BCs(b_size, x=0.0, u=0.0, deriv=0))
    bcs.append(BCs(b_size, x=0.6, u=0.0, deriv=0))
    bcs.append(BCs(b_size, x=1.0, u=0.0, deriv=0))
    bcs.append(BCs(b_size, x=0.0, u=0.0, deriv=2))
    bcs.append(BCs(b_size, x=1.0, u=0.0, deriv=2))
    bcs.append(BCs(b_size, x=0.6, u=0.0, deriv=1))

    optims = []
    schedulers = []

    # models = test._modules['module']._modules
    models = test._modules
    # print(models)

    for key in models.keys():
        model = models[key]
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
        optims.append(optim)
        schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=100, verbose=True))

    # dms = test.module.domains
    # bds = test.module.boundaries
    dms = test.domains
    bds = test.boundaries
    # Penalty term
    w = 1000

    x_bs = []
    u_bs = []
    x_fs = []
    u_fs = []

    x_derivs = []
    x_derivs_train = [[] for _ in range(domain_no)]

    x_bs_train = [[] for _ in range(domain_no)]
    u_bs_train = [[] for _ in range(domain_no)]
    
    for bc in bcs:
        x_b, u_b = make_training_boundary_data(b_size=bc.size, x=bc.x, u=bc.u)
        x_bs.append(x_b.to(device))
        u_bs.append(u_b.to(device))
        x_derivs.append(bc.deriv)

    for i, dm in enumerate(dms):
        lb = dm['lb']
        rb = dm['rb']
        # print(lb, rb)
        x_f, u_f = make_training_collocation_data(f_size, x_lb=lb, x_rb=rb)
        x_fs.append(x_f.to(device))
        u_fs.append(u_f.to(device))

        for j, x_b in enumerate(x_bs):
            u_b = u_bs[j]
            x_deriv = x_derivs[j]
            x = x_b[0]
            if lb <= x <= rb:
                x_bs_train[i].append(x_b)
                u_bs_train[i].append(u_b)
                x_derivs_train[i].append(x_deriv)

    loss_save = np.inf
    
    loss_b_plt = [[] for _ in range(domain_no)]
    loss_f_plt = [[] for _ in range(domain_no)]
    loss_plt   = [[] for _ in range(domain_no)]

    x_plt = torch.from_numpy(np.arange((global_rb - global_lb) * 1000) / 1000 + global_lb) 

    for epoch in range(epochs):
        for i in range(domain_no):
            optim = optims[i]
            scheduler = schedulers[i]
            optim.zero_grad()
            loss_b = torch.zeros(1).to(device)
            loss_f = torch.zeros(1).to(device)
            loss_func = nn.MSELoss()

            loss_sum = torch.zeros(1).to(device)

            x_bs = x_bs_train[i]
            u_bs = u_bs_train[i]
            x_derivs = x_derivs_train[i]

            for j, x_b in enumerate(x_bs):
                u_b = u_bs[j]
                x_deriv = x_derivs[j]
                loss_b += loss_func(calc_deriv(x_b, test(x_b), x_deriv), u_b) * w

            x_f = x_fs[i]
            u_f = u_fs[i]
            loss_f += loss_func(calc_deriv(x_f, test(x_f), 4) - 1, u_f)
          
            loss = loss_f + loss_b
            loss_sum += loss

            loss_b_plt[i].append(loss_b.item())
            loss_f_plt[i].append(loss_f.item())
            loss_plt[i].append(loss.item())
            
            loss.backward()

            optim.step()
            scheduler.step(loss)

            with torch.no_grad():
                test.eval()
            
            print("Epoch: {0} | LOSS: {1:.5f}".format(epoch+1, loss.item()))

            if epoch % 50 == 1:
                draw_convergence(epoch + 1, loss_b_plt[i], loss_f_plt[i], loss_plt[i], i)

        if loss_sum < loss_save:
            loss_save = loss_sum
            torch.save(test.state_dict(), './models/cpinn.data')
            print(".......model updated (epoch = ", epoch+1, ")")
        
        if loss_sum < 0.00001:
            break

        if epoch % 50 == 1:
            draw(domain_no, global_lb, global_rb, overlap_size)
            test.plot_separate_models(x_plt)
            
    print("Elapsed Time: {} s".format(time.time() - since))
    print("DONE")
    
def draw_convergence(epoch, loss_b, loss_f, loss, id):
    plt.cla()
    x = np.arange(epoch)
    # print(epoch)
    # print(loss_b)
    # print(loss_f)
    # print(loss)

    plt.plot(x, np.array(loss_b), label='Loss_B')
    plt.plot(x, np.array(loss_f), label='Loss_F')
    plt.plot(x, np.array(loss), label='Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig('./figures/convergence_{}.png'.format(id))

def draw(domain_no, lb, rb, overlap_size):
    model = CombinedPINN(domain_no, lb, rb, overlap_size)
    sample = {'Model{}'.format(i+1): PINN(i) for i in range(domain_no)}
    model.module_update(sample)
    model.make_domains()
    model.make_boundaries()
    model.cuda()
    model_path = "./models/cpinn.data"
    state_dict = torch.load(model_path)
    state_dict = {m.replace('module.', '') : i for m, i in state_dict.items()}
    model.load_state_dict(state_dict)

    x_test_plt = np.arange(10001)/10000
    x_test = torch.from_numpy(x_test_plt).unsqueeze(0).T.type(torch.FloatTensor).cuda()
    
    plt.cla()

    pred = model(x_test).cpu().detach().numpy()

    plt.plot(x_test_plt, pred, 'b', label='CPINN')
    # plt.plot(x_test, ex, 'r--', label='Exact')
    plt.legend()
    plt.savefig('./figures/test.png')

    plt.cla()
    
    # plt.plot(x_test_plt, model.module.out_A(x_test).cpu().detach().numpy(), 'b', label='A')
    # plt.plot(x_test_plt, model.module.out_B(x_test).cpu().detach().numpy(), 'r--', label='B')
    # plt.legend()
    # plt.savefig('./figures/separate.png')

    # plt.cla()
    # plt.plot(x_test_plt, window(x_test, 1/3, 2/3, i=0).cpu().detach().numpy(), 'b', label='A')
    # plt.plot(x_test_plt, window(x_test, 1/3, 2/3, i=1).cpu().detach().numpy(), 'r--', label='B')
    # plt.legend()
    # plt.savefig('./figures/window.png')





def main():
    train()
    # window_test()
    

if __name__ == "__main__":
    main()
    