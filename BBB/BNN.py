from BNNLayer import BNNLayer

import torch
import torch.nn as nn


class BNN(nn.Module):
    def __init__(self, n_input, hidden_size, n_output):
        super(BNN, self).__init__()
        self.input = BNNLayer(n_input,
                              hidden_size,
                              activation='relu',
                              prior_mean=0, prior_rho=-3)
        self.hidden = BNNLayer(hidden_size,
                               hidden_size,
                               activation='relu',
                               prior_mean=0, prior_rho=-3)
        self.output = BNNLayer(hidden_size,
                               n_output,
                               activation='softmax',
                               prior_mean=0, prior_rho=-3)

        # Put in a list for convenient iterating over the layers
        self.layers = []
        for layer in [self.input, self.hidden, self.output]: self.layers.append(layer)

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

    def Forward(self, x, y, n_samples):
        total_likelh = 0
        total_kl = 0

        # Sample N samples and average
        for _ in range(n_samples):
            out, kl = self.forward(x, mode='forward')
            likelh = torch.log(out.gather(1, y)).sum()

            total_kl += kl
            total_likelh += likelh

        return total_kl / n_samples, total_likelh / n_samples

    @staticmethod
    def loss_fn(kl, likelh, n_batch):
        return (kl / n_batch - likelh).mean()
    
