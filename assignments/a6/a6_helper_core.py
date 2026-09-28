# Core implementation adapted from University of Michigan EECS 498/598 course starter code.
# Static validation only; not executed.
import math
import os
import pickle
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as T
from scipy.ndimage.filters import gaussian_filter1d
from torch import nn
from torch.utils.data import DataLoader, sampler
from vae_core import loss_function

def hello_helper():
    print('Hello from a6_helper.py!')
SQUEEZENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float)
SQUEEZENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float)

def preprocess(img, size=224):
    transform = T.Compose([T.Resize(size), T.ToTensor(), T.Normalize(mean=SQUEEZENET_MEAN.tolist(), std=SQUEEZENET_STD.tolist()), T.Lambda(lambda x: x[None])])
    return transform(img)

def deprocess(img, should_rescale=True):
    transform = T.Compose([T.Lambda(lambda x: x[0]), T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()), T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]), T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x), T.ToPILImage()])
    return transform(img)

def rescale(x):
    low, high = (x.min(), x.max())
    x_rescaled = (x - low) / (high - low)
    return x_rescaled

def blur_image(X, sigma=1):
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X

def check_scipy():
    import scipy
    vnum = int(scipy.__version__.split('.')[1])
    major_vnum = int(scipy.__version__.split('.')[0])
    assert vnum >= 16 or major_vnum >= 1, 'You must install SciPy >= 0.16.0 to complete this notebook.'

def jitter(X, ox, oy):
    if ox != 0:
        left = X[:, :, :, :-ox]
        right = X[:, :, :, -ox:]
        X = torch.cat([right, left], dim=3)
    if oy != 0:
        top = X[:, :, :-oy]
        bottom = X[:, :, -oy:]
        X = torch.cat([bottom, top], dim=2)
    return X

def load_CIFAR(path='./datasets/'):
    NUM_TRAIN = 49000
    transform = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))])
    cifar10_train = dset.CIFAR10(path, train=True, download=True, transform=transform)
    loader_train = DataLoader(cifar10_train, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
    cifar10_val = dset.CIFAR10(path, train=True, download=True, transform=transform)
    loader_val = DataLoader(cifar10_val, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))
    cifar10_test = dset.CIFAR10(path, train=False, download=True, transform=transform)
    loader_test = DataLoader(cifar10_test, batch_size=64)
    return (loader_train, loader_val, loader_test)

def load_imagenet_val(num=None, path='./datasets/imagenet_val_25.npz'):
    imagenet_fn = os.path.join(path)
    if not os.path.isfile(imagenet_fn):
        print('file %s not found' % imagenet_fn)
        print('Run the above cell to download the data')
        assert False, 'Need to download imagenet_val_25.npz'
    f = np.load(imagenet_fn, allow_pickle=True)
    X = f['X']
    y = f['y']
    class_names = f['label_map'].item()
    if num is not None:
        X = X[:num]
        y = y[:num]
    return (X, y, class_names)

def load_COCO(path='./datasets/coco.pt'):
    data_dict = torch.load(path)
    for k, v in data_dict.items():
        if type(v) == torch.Tensor:
            print(k, type(v), v.shape, v.dtype)
        else:
            print(k, type(v), v.keys())
    num_train = data_dict['train_images'].size(0)
    num_val = data_dict['val_images'].size(0)
    assert data_dict['train_images'].size(0) == data_dict['train_captions'].size(0) and data_dict['val_images'].size(0) == data_dict['val_captions'].size(0), 'shapes of data mismatch!'
    print('\nTrain images shape: ', data_dict['train_images'].shape)
    print('Train caption tokens shape: ', data_dict['train_captions'].shape)
    print('Validation images shape: ', data_dict['val_images'].shape)
    print('Validation caption tokens shape: ', data_dict['val_captions'].shape)
    print('total number of caption tokens: ', len(data_dict['vocab']['idx_to_token']))
    print('mappings (list) from index to caption token: ', data_dict['vocab']['idx_to_token'])
    print('mappings (dict) from caption token to index: ', data_dict['vocab']['token_to_idx'])
    return data_dict

def dump_results(submission, path):
    with open(path, 'wb') as f:
        pickle.dump(submission, f)

def get_zero_one_masks(img, size):
    transform = T.Compose([T.Resize(size), T.ToTensor()])
    img = transform(img).sum(dim=0, keepdim=True)
    mask = torch.where(img < 1, 0, 1).to(torch.float)
    return mask

def show_images(images):
    images = torch.reshape(images, [images.shape[0], -1])
    sqrtn = int(math.ceil(math.sqrt(images.shape[0])))
    sqrtimg = int(math.ceil(math.sqrt(images.shape[1])))
    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))
    return

def count_params(model):
    param_count = sum([p.numel() for p in model.parameters()])
    return param_count

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight.data)

def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets

def train_vae(epoch, model, train_loader, cond=False):
    model.train()
    train_loss = 0
    num_classes = 10
    loss = None
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device='cuda:0')
        if cond:
            one_hot_vec = one_hot(labels, num_classes).to(device='cuda')
            recon_batch, mu, logvar = model(data, one_hot_vec)
        else:
            recon_batch, mu, logvar = model(data)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data
        optimizer.step()
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.data))
