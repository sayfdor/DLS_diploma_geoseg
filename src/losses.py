import torch
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()

    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce)
    loss = alpha * (1 - pt) ** gamma * bce
    return loss.mean()


def morphological_gradient(mask, kernel_size=3):

    device = mask.device
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)

    dilated = F.conv2d(mask.float(), kernel, padding=kernel_size // 2)
    dilated = (dilated > 0).float()

    eroded = F.conv2d(mask.float(), kernel, padding=kernel_size // 2)
    eroded = (eroded == kernel.sum()).float()

    gradient = dilated - eroded
    return gradient.clamp(min=0)


def combined_loss(pred, target, alpha=0.4, beta=0.4, gamma=0.2, kernel_size=3):

    dice = dice_loss(pred, target)
    focal = focal_loss(pred, target)

    boundary = morphological_gradient(target, kernel_size=kernel_size)
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    boundary_loss_val = (bce * boundary).sum() / (boundary.sum() + 1e-6)

    return alpha * dice + beta * focal + gamma * boundary_loss_val
