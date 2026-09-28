# Core implementation adapted from University of Michigan EECS 498/598 course starter code.
# Static validation only; not executed.
from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

def hello_vae():
    print('Hello from vae.py!')

class VAE(nn.Module):

    def __init__(self, input_size, latent_size=15):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_dim = None
        self.encoder = None
        self.mu_layer = None
        self.logvar_layer = None
        self.decoder = None
        self.hidden_dim = 400
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(input_size, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU())
        self.mu_layer = nn.Linear(self.hidden_dim, latent_size)
        self.logvar_layer = nn.Linear(self.hidden_dim, latent_size)
        side = int(input_size ** 0.5)
        self.decoder = nn.Sequential(nn.Linear(latent_size, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, input_size), nn.Sigmoid(), nn.Unflatten(1, (1, side, side)))

    def forward(self, x):
        x_hat = None
        mu = None
        logvar = None
        h = self.encoder(x)
        mu, logvar = (self.mu_layer(h), self.logvar_layer(h))
        x_hat = self.decoder(reparametrize(mu, logvar))
        return (x_hat, mu, logvar)

class CVAE(nn.Module):

    def __init__(self, input_size, num_classes=10, latent_size=15):
        super(CVAE, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.hidden_dim = None
        self.encoder = None
        self.mu_layer = None
        self.logvar_layer = None
        self.decoder = None
        self.hidden_dim = 400
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(input_size + num_classes, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU())
        self.mu_layer = nn.Linear(self.hidden_dim, latent_size)
        self.logvar_layer = nn.Linear(self.hidden_dim, latent_size)
        side = int(input_size ** 0.5)
        self.decoder = nn.Sequential(nn.Linear(latent_size + num_classes, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, input_size), nn.Sigmoid(), nn.Unflatten(1, (1, side, side)))

    def forward(self, x, c):
        x_hat = None
        mu = None
        logvar = None
        h = self.encoder(torch.cat((x.flatten(1), c), dim=1))
        mu, logvar = (self.mu_layer(h), self.logvar_layer(h))
        z = reparametrize(mu, logvar)
        x_hat = self.decoder(torch.cat((z, c), dim=1))
        return (x_hat, mu, logvar)

def reparametrize(mu, logvar):
    z = None
    z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
    return z

def loss_function(x_hat, x, mu, logvar):
    loss = None
    reconstruction = F.binary_cross_entropy(x_hat, x, reduction='sum')
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum()
    loss = (reconstruction + kl) / x.shape[0]
    return loss
