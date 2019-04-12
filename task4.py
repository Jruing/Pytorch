import torch
from torch.autograd import Variable
import numpy as np
class module_net(torch.nn.Module):
    def __init__(self, num_input, num_hidden, num_output):
        super(module_net, self).__init__()
        self.layer1 = torch.nn.Linear(num_input, num_hidden)           
        self.layer2 = torch.nn.Tanh()
        self.layer3 = torch.nn.Linear(num_hidden, num_output)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    
if __name__ == '__main__':
    xy=np.loadtxt('diabetes.csv.gz',delimiter='.'.dtype=np.float32)
    x=Variable(torch.from_numpy(xy[:,0:-1]))
    y=Variable(torch.from_numpy(xy[:,[-1]]))
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    mo_net = module_net(2, 4, 1)    
    optim = torch.optim.SGD(mo_net.parameters(), lr=0.0001)
    criterion = nn.BCEWithLogitsLoss()
    for e in range(10000):
        out = mo_net(Variable(x))
        loss = criterion(out, Variable(y))
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (e + 1) % 1000 == 0:
            print('epoch: {}, loss: {}'.format(e+1, loss.item()))