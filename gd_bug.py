import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TwoLayers(nn.Module):
    def __init__(self):
        super(TwoLayers, self).__init__()
        self.l1 = nn.Linear(1, 1, bias=False)
        self.l2 = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))

    def get_hidden(self, x):
        return F.relu(self.l1(x))


class OneLayer(nn.Module):
    def __init__(self):
        super(OneLayer, self).__init__()
        self.l1 = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        return self.l1(x)


def run():
    x, y = ([1, 2, 3], [1, -1, 2])
    x_t = torch.tensor([[i] for i in x]).float()
    y_t = torch.tensor([[i] for i in y]).float()

    net1 = OneLayer()
    net2 = TwoLayers()
    
    torch.nn.init.uniform_(net1.l1.weight, a=1, b=1)
    torch.nn.init.uniform_(net2.l1.weight, a=1, b=1)
    torch.nn.init.uniform_(net2.l2.weight, a=1, b=1)

    net2.l1.weight.requires_grad = False

    criterion = nn.MSELoss()
    opt1 = optim.SGD(net1.parameters(), lr=0.001)
    opt2 = optim.SGD(net2.parameters(), lr=0.001)

    hidden = net2.get_hidden(x_t)

    for epoch in range(10):
        opt1.zero_grad()
        out1 = net1(hidden)
        # print('out1: {}'.format(out1))
        loss1 = criterion(out1, y_t)
        print('loss1: {}'.format(loss1))
        loss1.backward()
        opt1.step()

        opt2.zero_grad()
        out2 = net2(x_t)
        # print('out2: {}'.format(out2))
        loss2 = criterion(out2, y_t)
        print('loss2: {}'.format(loss2))
        loss2.backward()
        opt2.step()


run()
