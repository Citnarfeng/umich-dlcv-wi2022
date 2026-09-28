# Core implementation adapted from University of Michigan EECS 498/598 course starter code.
# Static validation only; not executed.
from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
NOISE_DIM = 96

def hello_gan():
    print('Hello from gan.py!')

def sample_noise(batch_size, noise_dim, dtype=torch.float, device='cpu'):
    noise = None
    noise = 2 * torch.rand(batch_size, noise_dim, dtype=dtype, device=device) - 1
    return noise

def discriminator():
    model = None
    model = nn.Sequential(nn.Linear(784, 256), nn.LeakyReLU(0.01), nn.Linear(256, 256), nn.LeakyReLU(0.01), nn.Linear(256, 1))
    return model

def generator(noise_dim=NOISE_DIM):
    model = None
    model = nn.Sequential(nn.Linear(noise_dim, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 784), nn.Tanh())
    return model

def discriminator_loss(logits_real, logits_fake):
    loss = None
    loss = nn.functional.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real)) + nn.functional.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
    return loss

def generator_loss(logits_fake):
    loss = None
    loss = nn.functional.binary_cross_entropy_with_logits(logits_fake, torch.ones_like(logits_fake))
    return loss

def get_optimizer(model):
    optimizer = None
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
    return optimizer

def ls_discriminator_loss(scores_real, scores_fake):
    loss = None
    loss = 0.5 * ((scores_real - 1).square().mean() + scores_fake.square().mean())
    return loss

def ls_generator_loss(scores_fake):
    loss = None
    loss = 0.5 * (scores_fake - 1).square().mean()
    return loss

def build_dc_classifier():
    model = None
    model = nn.Sequential(nn.Unflatten(1, (1, 28, 28)), nn.Conv2d(1, 32, 5), nn.LeakyReLU(0.01), nn.MaxPool2d(2, 2), nn.Conv2d(32, 64, 5), nn.LeakyReLU(0.01), nn.MaxPool2d(2, 2), nn.Flatten(), nn.Linear(4 * 4 * 64, 4 * 4 * 64), nn.LeakyReLU(0.01), nn.Linear(4 * 4 * 64, 1))
    return model

def build_dc_generator(noise_dim=NOISE_DIM):
    model = None
    model = nn.Sequential(nn.Linear(noise_dim, 1024), nn.ReLU(), nn.BatchNorm1d(1024), nn.Linear(1024, 7 * 7 * 128), nn.ReLU(), nn.BatchNorm1d(7 * 7 * 128), nn.Unflatten(1, (128, 7, 7)), nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(), nn.BatchNorm2d(64), nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh(), nn.Flatten())
    return model
