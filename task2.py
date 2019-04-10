# encoding:utf8

from torch.autograd import Variable
import torch
torch.manual_seed(2)
x_data = Variable(torch.Tensor([[1.0],[2.0],[3.0],[4.0]]))
y_data = Variable(torch.Tensor([[0.0],[1.0],[2.0],[3.0]]))
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear = torch.nn.Linear(1,1)
    def forward(self,x):
        y = self.linear(x)
        return y
model = Model()
criterion = torch.nn.BCEWidthLogitsLoss()
optimizeer = torch.optim.SGD(model.parameters(),lr=0.01)
hour_var = Variable(torch.Tensor([[2.4]]))
y = model(hour_var)

float(model(hour_var).data[0][0]>0.5)
