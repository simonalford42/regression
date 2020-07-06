import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


def run():
    x, y = list(range(1, 11)), [1, 2, 3, 4, 5, 4, 30, 2, 1, 0]
    x2 = np.array([[i] for i in x])

    reg = LinearRegression().fit(x2, np.array(y))
    reg_prediction = reg.predict(x2)
    reg_residuals = residuals(y, reg_prediction)
    print('reg_residuals: {}'.format(reg_residuals))

    x_t = torch.tensor([[i] for i in x]).float()
    y_t = torch.tensor([[i] for i in y]).float()
    net = SmallNet(1)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)

    nn_residuals = []
    hidden_residuals = []
    epochs = []

    for epoch in range(250000):
        optimizer.zero_grad()
        outputs = net(x_t)
        loss = criterion(outputs, y_t)
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print('epoch: {}, loss: {}'.format(epoch, loss))

        if epoch % 500 == 0:
            nn_res, hidden_res = nn_and_hidden_residuals(net, x_t, y)
            nn_residuals.append(nn_res)
            hidden_residuals.append(hidden_res)
            epochs.append(epoch)

    plt.plot(epochs, nn_residuals, label='nn_residuals')
    plt.plot(epochs, hidden_residuals, label='hidden_residuals')
    plt.xlabel('epoch')
    plt.ylabel('residual')

    # plt.plot(x, y, 'bo')
    # plt.plot(x, nn_prediction, label='nn', alpha=0.7)
    # plt.plot(x, reg_prediction, label='regression', alpha=0.7)

    plt.legend()
    plt.show()


def residuals(y, y_hat):
    return sum([y_i_hat * (y_i - y_i_hat) for y_i, y_i_hat in zip(y, y_hat)])


def nn_and_hidden_residuals(net, x_t, y):
    nn_prediction = net(x_t).detach().numpy()
    nn_residuals = residuals(y, nn_prediction)

    hidden_values = net.get_hidden(x_t).detach().numpy()
    hidden_reg = LinearRegression().fit(hidden_values, np.array(y))
    hidden_reg_prediction = hidden_reg.predict(hidden_values)
    hidden_residuals = residuals(y, hidden_reg_prediction)

    return nn_residuals, hidden_residuals


class SmallNet(nn.Module):

    def __init__(self, input_dim):
        super(SmallNet, self).__init__()
        self.hidden_dim = 10
        self.hidden = nn.Linear(input_dim, self.hidden_dim)
        self.last = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        return self.last(x)

    def get_hidden(self, x):
        return F.relu(self.hidden(x))


def train(net, x, y):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(50000):
        optimizer.zero_grad()
        outputs = net(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print('epoch: {}, loss: {}'.format(epoch, loss))


run()
