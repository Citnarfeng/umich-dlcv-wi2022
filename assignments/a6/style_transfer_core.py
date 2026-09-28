# Core implementation adapted from University of Michigan EECS 498/598 course starter code.
# Static validation only; not executed.
import torch
import torch.nn as nn
from a6_helper_core import *

def hello():
    print('Hello from style_transfer.py!')

def content_loss(content_weight, content_current, content_original):
    return content_weight * (content_current - content_original).square().sum()

def gram_matrix(features, normalize=True):
    gram = None
    flat = features.flatten(2)
    gram = flat @ flat.transpose(1, 2)
    if normalize:
        gram = gram / (features.shape[1] * features.shape[2] * features.shape[3])
    return gram

def style_loss(feats, style_layers, style_targets, style_weights):
    loss = feats[0].new_zeros(())
    for layer, target, weight in zip(style_layers, style_targets, style_weights):
        loss = loss + weight * (gram_matrix(feats[layer]) - target).square().sum()
    return loss

def tv_loss(img, tv_weight):
    return tv_weight * ((img[:, :, 1:, :] - img[:, :, :-1, :]).square().sum() + (img[:, :, :, 1:] - img[:, :, :, :-1]).square().sum())

def guided_gram_matrix(features, masks, normalize=True):
    guided_gram = None
    flat = (features * masks.unsqueeze(2)).flatten(3)
    guided_gram = flat @ flat.transpose(-1, -2)
    if normalize:
        guided_gram = guided_gram / (features.shape[2] * features.shape[3] * features.shape[4])
    return guided_gram

def guided_style_loss(feats, style_layers, style_targets, style_weights, content_masks):
    loss = feats[0].new_zeros(())
    for layer, target, weight in zip(style_layers, style_targets, style_weights):
        features = feats[layer]
        if features.ndim == 4:
            features = features.unsqueeze(1)
        gram = guided_gram_matrix(features, content_masks[layer])
        loss = loss + weight * (gram - target).square().sum()
    return loss
