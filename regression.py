import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


def run():
    x, y = list(range(1, 11)), [1, 2, 3, 4, 5, 4, 30, 2, 1, 0]
    # plt.plot(x, y, 'bo')

    x_t = torch.tensor([[i] for i in x]).float()
    y_t = torch.tensor([[i] for i in y]).float()
    net = SmallNet(1)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)

    nn_residuals = []
    hidden_residuals = []
    hidden_weights = []
    epochs = []

    epoch = 0
    while epoch < 500000:
        epoch += 1
        optimizer.zero_grad()
        outputs = net(x_t)
        loss = criterion(outputs, y_t)
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print('epoch: {}, loss: {}'.format(epoch, loss))

        # may want to save with higher frequency if training less often
        # if epoch % 50 == 0:
        if epoch % 500 == 0:
            nn_prediction = net(x_t).detach().numpy()
            nn_res = residuals(y, nn_prediction)
            nn_residuals.append(nn_res)

            hidden_values = net.get_hidden(x_t).detach().numpy()
            hidden_weights.append(hidden_values.sum(axis=0))

            hidden_reg = LinearRegression().fit(hidden_values, np.array(y))
            hidden_reg_prediction = hidden_reg.predict(hidden_values)
            hidden_res = residuals(y, hidden_reg_prediction)
            hidden_residuals.append(hidden_res)

            epochs.append(epoch)

    # freeze the weights from the first layer
    for i, param in enumerate(net.parameters()):
        if i <= 1:
            param.requires_grad = False

    # continue training with the frozen first layer
    while epoch < 20000:
        epoch += 1
        optimizer.zero_grad()
        outputs = net(x_t)
        loss = criterion(outputs, y_t)
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print('epoch: {}, loss: {}'.format(epoch, loss))

        if epoch % 50 == 0:
            nn_prediction = net(x_t).detach().numpy()
            nn_res = residuals(y, nn_prediction)

            hidden_values = net.get_hidden(x_t).detach().numpy()
            hidden_reg = LinearRegression().fit(hidden_values, np.array(y))
            hidden_reg_prediction = hidden_reg.predict(hidden_values)
            hidden_res = residuals(y, hidden_reg_prediction)
            nn_residuals.append(nn_res)
            hidden_residuals.append(hidden_res)
            epochs.append(epoch)
            hidden_weights.append(hidden_values.sum(axis=0))
        
    hidden_weights = np.column_stack(hidden_weights)
    for i, h_w in enumerate(hidden_weights):
        plt.plot(epochs, h_w, label='Hidden node {}'.format(i))

    plt.plot(epochs, nn_residuals, label='nn_residuals')
    plt.plot(epochs, hidden_residuals, label='hidden_residuals')
    plt.xlabel('epoch')

    plt.legend()
    plt.show()


def residuals(y, y_hat):
    return sum([y_i_hat * (y_i - y_i_hat) for y_i, y_i_hat in zip(y, y_hat)])


class SmallNet(nn.Module):

    def __init__(self, input_dim):
        super(SmallNet, self).__init__()
        self.hidden_dim = 3
        self.hidden = nn.Linear(input_dim, self.hidden_dim)
        self.last = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        return self.last(x)

    def get_hidden(self, x):
        return F.relu(self.hidden(x))


run()
