import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.hidden_layer1      = nn.Linear(3, 20)
        self.hidden_layer2      = nn.Linear(20, 20)
        self.hidden_layer3      = nn.Linear(20, 20)
        self.hidden_layer4      = nn.Linear(20, 20)
        # self.hidden_layer5      = nn.Linear(20, 20)
        self.output_layer       = nn.Linear(20, 2)


    def forward(self, x, y, e):
        input_data     = torch.cat((x, y, e), axis=1)
        act_func       = nn.Tanh()
        a_layer1       = act_func(self.hidden_layer1(input_data))
        a_layer2       = act_func(self.hidden_layer2(a_layer1))
        a_layer3       = act_func(self.hidden_layer3(a_layer2))
        a_layer4       = act_func(self.hidden_layer4(a_layer3))
        # a_layer5       = act_func(self.hidden_layer5(a_layer4))
        out            = self.output_layer(a_layer4)

        return out

def train(model_path, figure_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current device:", device)
    
    model = MLP()
    model = model.to(device)
    epochs = 100000
    lr = 0.0001
    loss_func = nn.MSELoss()

    E = [2, 6, 10]
    data = [[] for _ in E]

    for i, e in enumerate(E):
        fname = "./data/surrogate/{}.txt".format(e)
        data[i] = pd.DataFrame(np.loadtxt(fname=fname), columns=['x', 'y', 'u', 'v'])
        data[i]['e'] = np.ones(data[i]['x'].shape) * e
     
    data = pd.concat(data)
    print(data)
    data_s = data.sample(n=500)
    # data_s = data
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_save = np.inf
    for epoch in range(epochs):
        optim.zero_grad()

        loss = 0.0

        x = torch.tensor(data_s['x'].values).unsqueeze(0).T.type(torch.FloatTensor).to(device)
        y = torch.tensor(data_s['y'].values).unsqueeze(0).T.type(torch.FloatTensor).to(device)
        e = torch.tensor(data_s['e'].values).unsqueeze(0).T.type(torch.FloatTensor).to(device)
        u = torch.tensor(data_s['u'].values).unsqueeze(0).T.type(torch.FloatTensor).to(device)
        v = torch.tensor(data_s['v'].values).unsqueeze(0).T.type(torch.FloatTensor).to(device)

        pred = model(x, y, e)
        
        loss = loss_func(pred, torch.cat((u, v), axis=1))
        loss.backward()
        optim.step()

        with torch.no_grad():
            model.eval()
            print("Epoch: {0} | LOSS: {1:.5f} ".format(epoch+1, loss.item()))

        if loss < loss_save:
            loss_save = loss
            torch.save(model.state_dict(), model_path)
            print(".......model updated (epoch = ", epoch+1, ")")

    print("DONE")
        

    

def main():
    train('./models/surrogate_FEM/test.data', None)

if __name__ == "__main__":
    main()
        
    
