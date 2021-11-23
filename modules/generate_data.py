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

def make_training_boundary_data_surrogate(b_size, x=0.0, u=0.0, seed=1004, x_b=None, u_b=None, device=None, w=None):
    np.random.seed(seed)

    x_b_new = np.ones((b_size, 1)) * x
    u_b_new = np.ones((b_size, 1)) * u
    e_b = np.ones((b_size, 1)) * w

    x_b_new = make_tensor(x_b_new)
    u_b_new = make_tensor(u_b_new)
    e_b = make_tensor(e_b)

    
    x_b = x_b_new
    u_b = u_b_new


    return x_b, u_b, e_b

def make_training_boundary_data_2d(size, x_lb, x_rb, y_lb, y_rb, u, v):
    if x_lb == x_rb:
        y_b =np.random.uniform(y_lb, y_rb, size=(size, 1))
        x_b = np.ones(y_b.shape) * x_lb
    elif y_lb == y_rb:
        x_b = np.random.uniform(x_lb, x_rb, size=(size, 1))
        y_b = np.ones(x_b.shape) * y_lb

    x_b = make_tensor(x_b)
    y_b = make_tensor(y_b)
    u_b = make_tensor(np.ones(x_b.shape)) * u
    v_b = make_tensor(np.ones(x_b.shape)) * v
    
    return x_b, y_b, u_b, v_b

def make_training_boundary_data_2d_surrogate(size, x_lb, x_rb, y_lb, y_rb, u, v, E):
    if x_lb == x_rb:
        y_b =np.random.uniform(y_lb, y_rb, size=(size, 1))
        x_b = np.ones(y_b.shape) * x_lb
    elif y_lb == y_rb:
        x_b = np.random.uniform(x_lb, x_rb, size=(size, 1))
        y_b = np.ones(x_b.shape) * y_lb

    x_b = make_tensor(x_b)
    y_b = make_tensor(y_b)
    u_b = make_tensor(np.ones(x_b.shape)) * u
    v_b = make_tensor(np.ones(x_b.shape)) * v
    e_b = make_tensor(np.ones(x_b.shape)) * E
    
    return x_b, y_b, u_b, v_b, e_b

def make_training_collocation_data(f_size, x_lb=0.0, x_rb=1.0, seed=1004, x_f=None, u_f=None):
    np.random.seed(seed)

    x_f_new = np.random.uniform(low=x_lb, high=x_rb, size=(f_size, 1))
    u_f_new = np.zeros((f_size, 1))

    x_f_new = make_tensor(x_f_new)
    u_f_new = make_tensor(u_f_new)

    x_f = torch.cat((x_f, x_f_new), axis=1) if x_f else x_f_new
    u_f = torch.cat((u_f, u_f_new), axis=1) if u_f else u_f_new

    return x_f, u_f

def make_training_collocation_data_surrogate(f_size, x_lb=0.0, x_rb=1.0, seed=1004, x_f=None, u_f=None, w=None):
    np.random.seed(seed)

    x_f_new = np.random.uniform(low=x_lb, high=x_rb, size=(f_size, 1))
    u_f_new = np.zeros((f_size, 1))
    e_f = np.ones((f_size, 1)) * w

    x_f_new = make_tensor(x_f_new)
    u_f_new = make_tensor(u_f_new)
    e_f = make_tensor(e_f)

    x_f = torch.cat((x_f, x_f_new), axis=1) if x_f else x_f_new
    u_f = torch.cat((u_f, u_f_new), axis=1) if u_f else u_f_new

    return x_f, u_f, e_f

def make_training_collocation_data_2d(size, x_lb, x_rb, y_lb, y_rb):
    x_f = make_tensor(np.random.uniform(x_lb, x_rb, size=(size, 1)))
    y_f = make_tensor(np.random.uniform(y_lb, y_rb, size=(size, 1)))
    u_f = make_tensor(np.zeros(x_f.shape))
    v_f = make_tensor(np.zeros(x_f.shape))

    return x_f, y_f, u_f, v_f

def make_training_collocation_data_2d_surrogate(size, x_lb, x_rb, y_lb, y_rb, E):
    x_f = make_tensor(np.random.uniform(x_lb, x_rb, size=(size, 1)))
    y_f = make_tensor(np.random.uniform(y_lb, y_rb, size=(size, 1)))
    u_f = make_tensor(np.zeros(x_f.shape))
    v_f = make_tensor(np.zeros(x_f.shape))
    e_f = make_tensor(np.ones(x_f.shape) * E)

    return x_f, y_f, u_f, v_f, e_f

def make_tensor(x, requires_grad=True):
    t = torch.from_numpy(x)
    t = t.float()

    t.requires_grad=requires_grad

    return t   

def to_device(*args, device):
    for x in args:
        x = x.to(device)
    return args

