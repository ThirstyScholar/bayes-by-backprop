"""
Adapt from:
1. https://gist.github.com/vvanirudh/9e30b2f908e801da1bd789f4ce3e7aac
2. "Variational Dropout and the Local Reparameterization Trick" (https://arxiv.org/abs/1506.02557)
3. http://gluon.mxnet.io/chapter18_variational-methods-and-uncertainty/bayes-by-backprop.html
"""

import numpy as np
import math

import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable

# Hyperparameters
N_Epochs = 15
N_Samples = 1
LearningRate = 1e-3
BatchSize = 100
Download_MNIST = False   # download the dataset if you don't already have it

training_set = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=Download_MNIST
)

N_Batch = training_set.train_data.size()[0] / BatchSize

train_loader = Data.DataLoader(dataset=training_set, batch_size=BatchSize, shuffle=True)

test_set = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=Download_MNIST
)


NegHalfLog2PI = -.5 * math.log(2.0 * math.pi)
softplus = lambda x: math.log(1 + math.exp(x))


def log_gaussian2(x, mean, std):
    return NegHalfLog2PI - torch.log(std) - .5 * torch.pow(x - mean, 2) / torch.pow(std, 2)


def sample_KL(x, mean1, std1, mean2, std2):
    log_prob1 = log_gaussian2(x, mean1, std1)
    log_prob2 = log_gaussian2(x, mean2, std2)
    return log_prob1 - log_prob2


class MLPLayer(nn.Module):
    def __init__(self, n_input, n_output, activation, prior_mean, prior_rho):
        assert activation == 'relu' or 'softmax' or 'none', 'Activation Type Not Found Error'

        super(MLPLayer, self).__init__()

        # Instantiate a large Gaussian block to sample from, much faster than generate random sample every time
        self._gaussian_block = np.random.randn(10000)

        self.n_input = n_input
        self.n_output = n_output

        # Hyperparameters for a layer
        self.W_mean = nn.Parameter(torch.ones((n_input, n_output)) * prior_mean)
        self.W_rho = nn.Parameter(torch.ones(n_input, n_output) * prior_rho)

        self.b_mean = nn.Parameter(torch.ones(1, n_output) * prior_mean)
        self.b_rho = nn.Parameter(torch.ones(1, n_output) * prior_rho)

        self.prior_var = Variable(torch.ones(1, 1) * softplus(prior_rho) ** 2)

        # Set activation func
        self.act = None
        if activation == 'relu':
            self.act = F.relu
        elif activation == 'softmax':
            self.act = F.softmax

        self._Var = lambda x: Variable(torch.from_numpy(x).type(torch.FloatTensor))

    def forward(self, X, mode):
        assert mode == 'forward' or 'MAP' or 'MC', 'MLPLayer Mode Not Found Error'

        _shape = (X.size()[0], self.n_output)

        # Z: pre-activation. Local reparam. trick is used.
        Z_Mean = torch.mm(X, self.W_mean) + self.b_mean.expand(*_shape)

        if mode == 'MAP': return self.act(Z_Mean) if self.act is not None else Z_Mean

        Z_Std = torch.sqrt(
            torch.mm(torch.pow(X, 2),
                     torch.pow(F.softplus(self.W_rho), 2)) +
            torch.pow(F.softplus(self.b_rho.expand(*_shape)), 2)
        )

        Z_noise = self._random(_shape)
        Z = Z_Mean + Z_Std * Z_noise

        if mode == 'MC': return self.act(Z) if self.act is not None else Z

        # Stddev for the prior
        Prior_Z_Std = torch.sqrt(
            torch.mm(torch.pow(X, 2),
                     self.prior_var.expand(self.n_input, self.n_output)) +
            self.prior_var.expand(*_shape)
        ).detach()

        # KL[posterior(w|D)||prior(w)]
        layer_KL = sample_KL(Z,
                             Z_Mean, Z_Std,
                             Z_Mean.detach(), Prior_Z_Std).sum()

        out = self.act(Z) if self.act is not None else Z
        return out, layer_KL

    def _random(self, shape):
        Z_noise = np.random.choice(self._gaussian_block, size=shape[0] * shape[1])
        Z_noise = np.expand_dims(Z_noise, axis=1).reshape(*shape)
        return self._Var(Z_noise)


class MLP(nn.Module):
    def __init__(self, n_input, hidden_size, n_output):
        super(MLP, self).__init__()
        self.input = MLPLayer(n_input, hidden_size, 'relu', prior_mean=0, prior_rho=-3)
        self.hidden = MLPLayer(hidden_size, hidden_size, 'relu', prior_mean=0, prior_rho=-3)
        self.output = MLPLayer(hidden_size, n_output, 'softmax', prior_mean=0, prior_rho=-3)

        self.layers = []
        for layer in [self.input, self.hidden, self.output]:
            self.layers.append(layer)

    def forward(self, x, mode):
        if mode == 'forward':
            net_kl = 0
            for layer in self.layers:
                x, layer_kl = layer.forward(x, mode)
                net_kl += layer_kl
            return x, net_kl
        else:
            for layer in self.layers:
                x = layer.forward(x, mode)
            return x


def Forward(net, x, y, n_samples):
    total_likelh = 0
    total_kl = 0
    for _ in range(n_samples):   # sample N samples and average
        out, kl = net.forward(x, mode='forward')

        # Likelihood of observing the data under the current weight configuration
        likelh = torch.log(out.gather(1, y)).sum()

        total_kl += kl
        total_likelh += likelh

    return total_kl / n_samples, total_likelh / n_samples


def loss_fn(kl, likelh, n_batch):
    return (kl / n_batch - likelh).mean()


# Initialize network
bnn = MLP(784, 128, 10)
optim = torch.optim.Adam(bnn.parameters(), lr=LearningRate)


# Main training loop
error_lst = []
for i_ep in range(N_Epochs):

    # Training
    for X, Y in train_loader:
        batch_X = Variable(X.view(X.size()[0], -1))
        batch_Y = Variable(Y.view(Y.size()[0], -1))

        kl, log_likelihood = Forward(bnn, batch_X, batch_Y, N_Samples)

        # Loss and backprop
        loss = loss_fn(kl, log_likelihood, N_Batch)
        optim.zero_grad()
        loss.backward()
        optim.step()

    # Evaluate on test set
    test_X = Variable(test_set.test_data.view(test_set.test_data.size()[0], -1).type(torch.FloatTensor))
    test_Y = Variable(test_set.test_labels.view(test_set.test_labels.size()[0], -1))

    pred_class = bnn.forward(test_X, mode='MAP').data.numpy().argmax(axis=1)
    true_class = test_Y.data.numpy().ravel()

    test_accu = (pred_class == true_class).mean()
    print('Epoch', i_ep, '|  Test Accuracy:', test_accu * 100, '%')

    error_lst.append((1 - test_accu) * 100)


# Plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')

plt.title('Test Error on MNIST')
plt.plot(error_lst)
plt.ylabel('Test error (%)')
plt.xlabel('Epochs')
plt.tight_layout()
plt.show()
