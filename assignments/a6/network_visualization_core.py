# Core implementation adapted from University of Michigan EECS 498/598 course starter code.
# Static validation only; not executed.
import torch

def hello():
    print('Hello from network_visualization.py!')

def compute_saliency_maps(X, y, model):
    X.requires_grad_()
    saliency = None
    scores = model(X).gather(1, y.reshape(-1, 1)).sum()
    grad = torch.autograd.grad(scores, X)[0]
    saliency = grad.abs().max(dim=1).values
    return saliency

def make_adversarial_attack(X, target_y, model, max_iter=100, verbose=True):
    X_adv = X.clone()
    X_adv = X_adv.requires_grad_()
    learning_rate = 1
    for iteration in range(max_iter):
        scores = model(X_adv)
        if scores.argmax(dim=1).item() == target_y:
            break
        grad = torch.autograd.grad(scores[0, target_y], X_adv)[0]
        norm = grad.norm()
        if norm.item() == 0:
            break
        with torch.no_grad():
            X_adv.add_(learning_rate * grad / norm)
        if verbose:
            print(f'Iteration {iteration}: target score {scores[0, target_y].item():.4f}')
    return X_adv

def class_visualization_step(img, target_y, model, **kwargs):
    l2_reg = kwargs.pop('l2_reg', 0.001)
    learning_rate = kwargs.pop('learning_rate', 25)
    score = model(img)[:, target_y].sum() - l2_reg * img.square().sum()
    grad = torch.autograd.grad(score, img)[0]
    with torch.no_grad():
        img.add_(learning_rate * grad)
    if img.grad is not None:
        img.grad.zero_()
    return img
