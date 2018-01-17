"""
Adapt from:
1. https://gist.github.com/vvanirudh/9e30b2f908e801da1bd789f4ce3e7aac
2. http://gluon.mxnet.io/chapter18_variational-methods-and-uncertainty/bayes-by-backprop.html
"""

# from copy import deepcopy
import numpy as np
import math

import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

import pdb


# Hyperparameters
N_Epochs = 15
N_Samples = 1
LearningRate = 1e-3
BatchSize = 100
Download_MNIST = False   # download the dataset if you don't already have it

# Gaussian prior
SigmaPrior = .05

# Scaled mixture Gaussian prior
Pi = .25
SigmaPrior1 = 2.
SigmaPrior2 = .1

training_set = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=Download_MNIST
)

N_Batch = training_set.train_data.size()[0] / BatchSize


# # Split training and validation set manually
# val_set = deepcopy(training_set)
#
# indices = list(range(len(training_set)))
# split = 50000
#
# train_idx = indices[:split]
#
# train_sampler = SubsetRandomSampler(train_idx)
# train_loader = torch.utils.data.DataLoader(training_set,
#                                            batch_size=BatchSize,
#                                            sampler=train_sampler)
#
# valid_sampler = SubsetRandomSampler(valid_idx)
# valid_loader = torch.utils.data.DataLoader(val_set,
#                                            batch_size=10000,
#                                            sampler=valid_sampler)
#
# val_X = training_set.train_data[split:]
# val_Y = training_set.train_labels[split:]

train_loader = Data.DataLoader(dataset=training_set, batch_size=BatchSize, shuffle=True)


test_set = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=Download_MNIST
)


NegHalfLog2PI = -.5 * math.log(2.0 * math.pi)


def log_gaussian(x, mu, sigma):
    """
    Log prob of a Gaussian
    :param x:
    :param mu:
    :param sigma: a real number
    :return: Log-Gaussian prob of sample x given mean mu and std sigam
    """
    if type(sigma) == Variable:
        return NegHalfLog2PI - torch.log(sigma) - (x - mu) ** 2 / (2 * sigma ** 2)
    else:
        return NegHalfLog2PI - math.log(sigma) - (x - mu) ** 2 / (2 * sigma ** 2)


def gaussian(x, mu, sigma):
    """
    Prob of a Gaussian
    :param x:
    :param mu:
    :param sigma:
    :return:
    """
    scaling = 1. / math.sqrt(2. * math.pi * (sigma ** 2))   # normalizing constant
    bell = torch.exp(- (x - mu) ** 2 / (2. * sigma ** 2))
    return scaling * bell


def log_mixture_gaussian(x):
    """
    Scaled mixture Gaussian prior
    :param x:
    :return:
    """
    gaussian1 = gaussian(x, 0, SigmaPrior1)
    gaussian2 = gaussian(x, 0, SigmaPrior2)
    return torch.log(Pi * gaussian1 + (1 - Pi) * gaussian2)


class MLPLayer(nn.Module):
    def __init__(self, n_input, n_output):
        super(MLPLayer, self).__init__()

        # Instantiate a large Gaussian block to sample from, much faster than generate random sample every time
        self._gaussian_block = np.random.randn(10000)

        self.n_input = n_input
        self.n_output = n_output

        # Hyperparameters for a layer
        match_prior = math.log(math.exp(SigmaPrior) - 1)

        self.W_mu = nn.Parameter(torch.FloatTensor(n_input, n_output).zero_())
        self.W_rho = nn.Parameter(torch.ones(n_input, n_output) * match_prior)

        self.b_mu = nn.Parameter(torch.FloatTensor(n_output).zero_())
        self.b_rho = nn.Parameter(torch.ones(n_output) * match_prior)

        self.log_prior = .0
        self.log_posterior = .0

        self._Var = lambda x: Variable(torch.from_numpy(x).type(torch.FloatTensor))

    def forward(self, X):
        epsilon_W, epsilon_b = self._random()

        W_sigma = F.softplus(self.W_rho)
        b_sigma = F.softplus(self.b_rho)

        # Construct weight matrix and bias by shifting it by a mean and scale it by a standard deviation
        W = self.W_mu + W_sigma * epsilon_W
        b = self.b_mu + b_sigma * epsilon_b

        # Compute output
        output = torch.mm(X, W) + b.expand(X.size()[0], W.size()[1])

        # Compute prior:
        # Gaussian prior
        log_prior = log_gaussian(W, 0, SigmaPrior).sum() + \
                    log_gaussian(b, 0, SigmaPrior).sum()

        # Mixture Gaussian prior
        # log_prior = log_mixture_gaussian(W).sum() + \
        #             log_mixture_gaussian(b).sum()

        # Compute posterior
        log_posterior = log_gaussian(W, self.W_mu, W_sigma).sum() + \
                        log_gaussian(b, self.b_mu, b_sigma).sum()

        return output, log_prior, log_posterior

    def infer_MAP(self, X):
        """
        MAP inference
        :param X:
        :return:
        """
        output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.W_mu.size()[1])
        return output

    def infer_MC(self, X):
        epsilon_W, epsilon_b = self._random()

        W_sigma = F.softplus(self.W_rho)
        b_sigma = F.softplus(self.b_rho)

        # Construct weight matrix and bias by shifting it by a mean and scale it by a standard deviation
        W = self.W_mu + W_sigma * epsilon_W
        b = self.b_mu + b_sigma * epsilon_b

        # Compute output
        output = torch.mm(X, W) + b.expand(X.size()[0], W.size()[1])

        return output

    def _random(self):
        W_noise = np.random.choice(self._gaussian_block, size=self.n_input * self.n_output)
        W_noise = np.expand_dims(W_noise, axis=1).reshape(self.n_input, self.n_output)
        W_noise = self._Var(W_noise)

        b_noise = np.random.choice(self._gaussian_block, size=self.n_output)
        b_noise = self._Var(b_noise)

        return W_noise, b_noise


class MLP(nn.Module):
    def __init__(self, n_input, hidden_size, n_output):
        super(MLP, self).__init__()

        self.input = MLPLayer(n_input, hidden_size)
        self.hidden = MLPLayer(hidden_size, hidden_size)
        self.output = MLPLayer(hidden_size, n_output)

        self.layers = []
        for layer in [self.input, self.hidden, self.output]:
            self.layers.append(layer)

    def forward(self, x):
        _log_prior = 0
        _log_posterior = 0

        for i, layer in enumerate(self.layers):
            x, layer_log_prior, layer_log_posterior = layer.forward(x)

            # Apply activation func
            if i != len(self.layers) - 1:
                x = F.relu(x)
            else:
                x = F.softmax(x)

            # Compute prior and posterior along the way
            _log_prior += layer_log_prior
            _log_posterior += layer_log_posterior

        return x, _log_prior, _log_posterior

    def infer(self, x, mode='MAP'):
        assert mode == 'MAP' or 'MC', 'Mode Not Found Error'

        for i, layer in enumerate(self.layers):

            if mode == 'MAP':
                x = layer.infer_MAP(x)
            else:   # mode == 'MC':
                x = layer.infer_MC(x)

            if i != len(self.layers) - 1:
                x = F.relu(x)
            else:
                x = F.softmax(x)
        return x

    def infer_MC(self, x):
        for i, layer in enumerate(self.layers):
            x = layer.infer_MC(x)
            if i != len(self.layers) - 1:
                x = F.relu(x)
            else:
                x = F.softmax(x)
        return x

    def infer_MAP(self, x):
        for i, layer in enumerate(self.layers):
            x = layer.infer_MAP(x)
            if i != len(self.layers) - 1:
                x = F.relu(x)
            else:
                x = F.softmax(x)
        return x


def Forward(X, Y):
    """
    :param X: batched
    :param Y: batched
    :return:
    """
    log_prior = 0
    log_posterior = 0
    log_likelihood = 0

    for _ in range(N_Samples):   # sample N samples and average

        # Forward and compute log prob
        output, sample_log_prior, sample_log_posterior = hyper_net.forward(X)

        # The likelihood of observing the data under the current weight configuration
        log_prob = torch.log(output.gather(1, Y))
        sample_log_likelihood = log_prob.sum()

        log_prior += sample_log_prior
        log_posterior += sample_log_posterior
        log_likelihood += sample_log_likelihood

    return log_prior / N_Samples, log_posterior / N_Samples, log_likelihood / N_Samples


def loss_fn(log_prior, log_posterior, log_likelihood):
    """
    The objective (loss) function 'f' in the paper given by (8)
    :param log_prior:
    :param log_posterior:
    :param log_likelihood:
    :return:
    """
    return ((1 / N_Batch) * (log_posterior - log_prior) - log_likelihood).sum() / BatchSize


# Initialize network
hyper_net = MLP(784, 128, 10)
optim = torch.optim.Adam(hyper_net.parameters(), lr=LearningRate)


# Main training loop
error_lst = []
for i_ep in range(N_Epochs):

    # Training
    for X, Y in train_loader:
        batch_X = Variable(X.view(X.size()[0], -1))
        batch_Y = Variable(Y.view(Y.size()[0], -1))

        log_prior, log_posterior, log_likelihood = Forward(batch_X, batch_Y)

        # Loss and backprop
        loss = loss_fn(log_prior, log_posterior, log_likelihood)
        optim.zero_grad()
        loss.backward()
        optim.step()


    # Evaluate on test set
    test_X = Variable(test_set.test_data.view(test_set.test_data.size()[0], -1).type(torch.FloatTensor))
    test_Y = Variable(test_set.test_labels.view(test_set.test_labels.size()[0], -1))

    pred_class = hyper_net.infer(test_X, mode='MAP').data.numpy().argmax(axis=1)
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
