import torch
import numpy as np
import torch.autograd as autograd

def make_training_boundary_data(b_size, x=0.0, u=0.0, seed=1004, x_b=None, u_b=None, device=None):
    np.random.seed(seed)

    x_b_new = np.ones((b_size, 1)) * x
    u_b_new = np.ones((b_size, 1)) * u

    x_b_new = make_tensor(x_b_new)
    u_b_new = make_tensor(u_b_new)

    
    x_b = torch.cat((x_b, x_b_new), axis=1) if x_b else x_b_new
    u_b = torch.cat((u_b, u_b_new), axis=1) if u_b else u_b_new


    return x_b, u_b

def make_training_collocation_data(f_size, x_lb=0.0, x_rb=1.0, seed=1004, x_f=None, u_f=None):
    np.random.seed(seed)

    x_f_new = np.random.uniform(low=x_lb, high=x_rb, size=(f_size, 1))
    u_f_new = np.zeros((f_size, 1))

    x_f_new = make_tensor(x_f_new)
    u_f_new = make_tensor(u_f_new)

    x_f = torch.cat((x_f, x_f_new), axis=1) if x_f else x_f_new
    u_f = torch.cat((u_f, u_f_new), axis=1) if u_f else u_f_new

    return x_f, u_f

def make_tensor(x, requires_grad=True):
    t = torch.from_numpy(x)
    t = t.float()

    t.requires_grad=requires_grad

    return t   

def to_device(*args, device):
    for x in args:
        x = x.to(device)
    return args

def calc_deriv(x, input, times):
    if times == 0:
        return input
    res = input
    for _ in range(times):
        res = autograd.grad(res.sum(), x, create_graph=True)[0]
    return res
