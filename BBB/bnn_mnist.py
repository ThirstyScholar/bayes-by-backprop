"""
Adapt from:
1. https://gist.github.com/vvanirudh/9e30b2f908e801da1bd789f4ce3e7aac
2. "Variational Dropout and the Local Reparameterization Trick" (https://arxiv.org/abs/1506.02557)
3. http://gluon.mxnet.io/chapter18_variational-methods-and-uncertainty/bayes-by-backprop.html
"""
from BNN import BNN

import torch
import torchvision
import torch.utils.data as Data
from torch.autograd import Variable

N_Epochs = 15
N_Samples = 1
LearningRate = 1e-3
BatchSize = 100
Download_MNIST = False   # download the dataset if you don't already have it

# Change to whatever directory your data is at
import os.path
dataset_path = os.path.join(os.path.dirname(__file__), 'mnist')

train_set = torchvision.datasets.MNIST(
    root=dataset_path,
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=Download_MNIST
)

train_size = train_set.train_data.size()[0]
N_Batch = train_size / BatchSize

train_loader = Data.DataLoader(dataset=train_set, batch_size=BatchSize, shuffle=True)

test_set = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=Download_MNIST
)

test_size = test_set.test_data.size()[0]

compute_accu = lambda pred, true, digits: round((pred == true).mean() * 100, digits)

if __name__ == '__main__':

    # Initialize network
    bnn = BNN(784, 128, 10)
    optim = torch.optim.Adam(bnn.parameters(), lr=LearningRate)

    # Main training loop
    train_accu_lst = []
    test_accu_lst = []
    for i_ep in range(N_Epochs):

        # Training
        for X, Y in train_loader:
            batch_X = Variable(X.view(BatchSize, -1))
            batch_Y = Variable(Y.view(BatchSize, -1))

            kl, log_likelihood = bnn.Forward(batch_X, batch_Y, N_Samples)

            # Loss and backprop
            loss = BNN.loss_fn(kl, log_likelihood, N_Batch)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # Evaluate on training set
        train_X = Variable(train_set.train_data.view(train_size, -1).type(torch.FloatTensor))
        train_Y = Variable(train_set.train_labels.view(train_size, -1))

        pred_class = bnn.forward(train_X, mode='MAP').data.numpy().argmax(axis=1)
        true_class = train_Y.data.numpy().ravel()

        train_accu = compute_accu(pred_class, true_class, 1)
        print('Epoch', i_ep, '|  Training Accuracy:', train_accu, '%')

        train_accu_lst.append(train_accu)

        # Evaluate on test set
        test_X = Variable(test_set.test_data.view(test_size, -1).type(torch.FloatTensor))
        test_Y = Variable(test_set.test_labels.view(test_size, -1))

        pred_class = bnn.forward(test_X, mode='MAP').data.numpy().argmax(axis=1)
        true_class = test_Y.data.numpy().ravel()

        test_accu = compute_accu(pred_class, true_class, 1)
        print('Epoch', i_ep, '|  Test Accuracy:', test_accu, '%')

        test_accu_lst.append(test_accu)

    # Plot
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-paper')

    plt.title('Classification Accuracy on MNIST')
    plt.plot(train_accu_lst, label='Train')
    plt.plot(test_accu_lst, label='Test')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epochs')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
