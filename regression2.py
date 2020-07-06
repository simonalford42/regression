
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def run():
    x, y = list(range(1, 11)), [1, 2, 3, 4, 5, 4, 30, 2, 1, 0]

    x_t = torch.tensor([[i] for i in x]).float()
    y_t = torch.tensor([[i] for i in y]).float()

    net = SmallNet(1)
    torch.nn.init.uniform_(net.hidden.weight, a=1, b=1)
    torch.nn.init.uniform_(net.last.weight, a=1, b=1)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)

    for i, param in enumerate(net.parameters()):
        print('i: {}'.format(i))
        print('param: {}'.format(param))
        if i == 0:
            param.requires_grad = False


    hidden_values = net.get_hidden(x_t)
    print('hidden_values: {}'.format(hidden_values))
    net2 = LinearNet(10)
    torch.nn.init.uniform_(net2.last.weight, a=1, b=1)
    optimizer2 = optim.SGD(net2.parameters(), lr=0.001)

    for i, param in enumerate(net2.parameters()):
        print('param: {}'.format(param))

    for epoch in range(10):
        optimizer.zero_grad()
        outputs = net(x_t)
        loss = criterion(outputs, y_t)
        loss.backward()
        optimizer.step()

        if epoch == 0:
            print('outputs: {}'.format(outputs))

        if epoch % 1 == 0:
            print('epoch: {}, loss: {}'.format(epoch, loss))

    for epoch in range(10):
        optimizer2.zero_grad()
        outputs = net2(hidden_values)
        loss = criterion(outputs, y_t)
        loss.backward()
        optimizer2.step()
        if epoch == 0:
            print('outputs: {}'.format(outputs))

        if epoch % 1 == 0:
            print('epoch: {}, loss: {}'.format(epoch, loss))

        

class SmallNet(nn.Module):

    def __init__(self, input_dim):
        super(SmallNet, self).__init__()
        self.hidden_dim = 10
        self.hidden = nn.Linear(input_dim, self.hidden_dim, bias=False)
        self.last = nn.Linear(self.hidden_dim, 1, bias=False)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        return self.last(x)

    def get_hidden(self, x):
        return F.relu(self.hidden(x))


class LinearNet(nn.Module):

    def __init__(self, input_dim):
        super(LinearNet, self).__init__()
        self.last = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        return self.last(x)

    def get_hidden(self, x):
        return F.relu(self.hidden(x))


run()