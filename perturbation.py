import torch
import random
from monai.losses import DiceLoss
import torch.nn as nn
import os

try:
    from .augmentation import RandomLabelAugmentation
except ImportError:  # allow running without package context
    from augmentation import RandomLabelAugmentation
from models.segmentation import Unet3DSeg

criterion = DiceLoss(include_background=False, to_onehot_y=True, softmax=True, squared_pred=True)

# bce_loss = nn.BCELoss()
bce_loss = nn.BCEWithLogitsLoss()

# sigmoid_fn = nn.Sigmoid()
softmax_fn = nn.Softmax(dim=1)

# random label aug
label_augmentation = RandomLabelAugmentation(device='cuda', p=0.5)


def transform_adv_prob_detach(input, mask, model, delta):

    # transform the output to probability map using sigmoid

    # Get the gradient of the model
    input = input.detach().clone()
    mask = mask.detach().clone()

    # print('input:', input.shape)

    input.requires_grad = True
    y_pred = model(input)

    # print('y_pred:', y_pred.shape)
    # print('mask:', mask.shape)

    loss = criterion(y_pred, mask)

    loss.backward()

    input_adv_noise = torch.sign(input.grad) * delta

    input_adv = input + input_adv_noise

    input.requires_grad = False

    y_pred_adv = model(input_adv)

    # print('diff', torch.max(torch.abs(y_pred_adv - y_pred)))

    loss_adv = criterion(y_pred_adv, mask)
    # print('after adv loss value:', loss_adv)

    return softmax_fn(y_pred_adv.detach()).clone()


def transform_adversarial_batch(inputs, masks, model, delta, num_classes=3):
    # Get the batch size
    B = masks.shape[0]
    
    # Determine the indices for the half batch to augment
    indices_to_augment = random.sample(range(B), B // 2)
    
    augmented_masks = nn.functional.one_hot(masks.long().squeeze(1), num_classes).permute(0, 3, 1, 2).float()

    # for idx in range(B):
    for idx in indices_to_augment:
        # if transform_type == 'adv':
        # print(inputs[idx].unsqueeze(0).shape, masks[idx].unsqueeze(0).shape)
        augmented_masks[idx] = transform_adv_prob_detach(inputs[idx].unsqueeze(0), masks[idx].unsqueeze(0), model, delta)

    return augmented_masks