import matplotlib.pyplot as plt
import numpy as np

from BNNLayer import BNNLayer
from BNN import BNN

import torch
from torch.autograd import Variable

plt.style.use('seaborn-paper')

x = np.random.uniform(-4, 4, size=20).reshape((-1, 1))
noise = np.random.normal(0, 9, size=20).reshape((-1, 1))
y = x ** 3 + noise

Var = lambda x, dtype=torch.FloatTensor: Variable(torch.from_numpy(x).type(dtype))
X = Var(x)
Y = Var(y)


if __name__ == '__main__':

    # Initialize network
    bnn = BNN(BNNLayer(1, 100, activation='relu', prior_mean=0, prior_rho=0),
              BNNLayer(100, 1, activation='none', prior_mean=0, prior_rho=0))

    optim = torch.optim.Adam(bnn.parameters(), lr=1e-1)

    # Main training loop
    for i_ep in range(400):
        kl, lg_lklh = bnn.Forward(X, Y, 1, 'Gaussian')
        loss = BNN.loss_fn(kl, lg_lklh, 1)
        optim.zero_grad()
        loss.backward()
        optim.step()

    # Plotting
    plt.scatter(x, y, c='navy', label='target')

    x_ = np.linspace(-5, 5)
    y_ = x_ ** 3
    X_ = Var(x_).unsqueeze(1)

    pred_lst = [bnn.forward(X_, mode='MC').data.numpy().squeeze(1) for _ in range(100)]

    pred = np.array(pred_lst).T
    pred_mean = pred.mean(axis=1)
    pred_std = pred.std(axis=1)

    plt.plot(x_, pred_mean, c='royalblue', label='mean pred')
    plt.fill_between(x_, pred_mean - 3 * pred_std, pred_mean + 3 * pred_std,
                     color='cornflowerblue', alpha=.5, label='+/- 3 std')

    plt.plot(x_, y_, c='grey', label='truth')

    plt.legend()
    plt.tight_layout()
    plt.show()
