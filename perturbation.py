import torch
import random
from monai.losses import DiceLoss

try:
    from augmentation import nonuniform_label_smooth, RandomLabelAugmentation
except:
    from .augmentation import nonuniform_label_smooth, RandomLabelAugmentation

import torch.nn as nn

import os
try:
    from networks import Unet3DSeg
except:
    from .networks import Unet3DSeg

criterion = DiceLoss(include_background=False, to_onehot_y=True, softmax=True, squared_pred=True)

# bce_loss = nn.BCELoss()
bce_loss = nn.BCEWithLogitsLoss()

# sigmoid_fn = nn.Sigmoid()
softmax_fn = nn.Softmax(dim=1)

# unfold = nn.Unfold(kernel_size=(16, 16), stride=16)
# fold = nn.Fold(output_size=(256, 256), kernel_size=(16, 16), stride=16)

# random label aug
label_augmentation = RandomLabelAugmentation(device='cuda', p=0.5)

'''
def transform_adv(input, mask, model, delta):
    # Get the gradient of the model
    input = input.clone()
    mask = mask.clone()

    # print('input:', input.shape)

    input.requires_grad = True
    y_pred = model(input)

    # print('y_pred:', y_pred.shape)
    # print('mask:', mask.shape)

    loss = criterion(y_pred, mask)

    loss.backward()

    # print('before adv loss value:', loss)

    input_adv_noise = torch.sign(input.grad) * delta

    input_adv = input + input_adv_noise

    input.requires_grad = False

    y_pred_adv = model(input_adv)

    # loss_adv = criterion(y_pred_adv, mask)
    # print('after adv loss value:', loss_adv)
    
    return y_pred_adv.detach().argmax(dim=1).clone()
'''

def adversarial_attack(input, label, model, delta):
    # if not input.requires_grad:
    #     raise ValueError("Input tensor must have requires_grad=True for gradient computation.")


    input = input.clone()
    label = label.clone()

    # input.requires_grad = True

    input.retain_grad()

    y_pred = model(input)

    loss = bce_loss(y_pred, label)

    model.zero_grad()

    loss.backward()

    input_adv_noise = torch.sign(input.grad) * delta

    input_adv = input + input_adv_noise

    return input_adv.detach().clone()

'''
def transform_adv_prob(input, mask, model, delta):

    # transform the output to probability map using sigmoid

    # Get the gradient of the model
    input = input.clone()
    mask = mask.clone()

    # print('input:', input.shape)

    input.requires_grad = True
    y_pred = model(input)

    # print('y_pred:', y_pred.shape)
    # print('mask:', mask.shape)

    loss = criterion(y_pred, mask)

    loss.backward()

    # print('before adv loss value:', loss)

    input_adv_noise = torch.sign(input.grad) * delta

    input_adv = input + input_adv_noise

    input.requires_grad = False

    y_pred_adv = model(input_adv)

    # loss_adv = criterion(y_pred_adv, mask)
    # print('after adv loss value:', loss_adv)
    
    # return sigmoid_fn(y_pred_adv).clone()
    return softmax_fn(y_pred_adv).clone()
'''

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

    # print('before adv loss value:', loss)

    input_adv_noise = torch.sign(input.grad) * delta

    input_adv = input + input_adv_noise

    input.requires_grad = False

    y_pred_adv = model(input_adv)

    print('diff', torch.max(torch.abs(y_pred_adv - y_pred)))

    loss_adv = criterion(y_pred_adv, mask)
    # print('after adv loss value:', loss_adv)
    
    # return sigmoid_fn(y_pred_adv).clone()
    return softmax_fn(y_pred_adv.detach()).clone()

'''
def transform_normal(input, mask, model, delta):
    input = input.clone()

    y_pred = model(input)

    return y_pred.detach().argmax(dim=1).clone()
'''

'''
def perturb_half_batch(inputs, masks, labels, model, delta, transform_type):
    assert transform_type in ['adv', 'normal']
    
    # Get the batch size
    B = masks.shape[0]
    
    # Determine the indices for the half batch to augment
    indices_to_augment = random.sample(range(B), B // 2)
    
    # Create a copy of masks and labels
    augmented_masks = masks.clone()
    augmented_labels = labels.clone()
    
    # Apply the augmentation
    for idx in indices_to_augment:
        if transform_type == 'adv':
            # print(inputs[idx].unsqueeze(0).shape, masks[idx].unsqueeze(0).shape)
            augmented_masks[idx] = transform_adv(inputs[idx].unsqueeze(0), masks[idx].unsqueeze(0), model, delta)
        elif transform_type == 'normal':
            augmented_masks[idx] = transform_normal(inputs[idx].unsqueeze(0), masks[idx].unsqueeze(0), model, delta)
        # augmented_masks[idx] = transform(masks[idx])
        augmented_labels[idx] = 0  # Set the corresponding label to 0
    
    return augmented_masks, augmented_labels


def perturb_half_batch_prob(inputs, masks, labels, model, delta, transform_type, num_classes, label_smoothing=1, prob_to_smooth=0.5):

    assert transform_type in ['adv']
    
    # Get the batch size
    B = masks.shape[0]
    
    # Determine the indices for the half batch to augment
    indices_to_augment = random.sample(range(B), B // 2)

    # print('indices_to_augment:', indices_to_augment)
    
    # Create a copy of masks and labels
    # augmented_masks = masks.clone()
    augmented_masks = torch.zeros((B, num_classes, masks.shape[2], masks.shape[3])).to(masks.device)
    augmented_labels = labels.clone()
    
    # Apply the 
    
    for idx in range(B):
        if idx in indices_to_augment:
            if transform_type == 'adv':
                # print(inputs[idx].unsqueeze(0).shape, masks[idx].unsqueeze(0).shape)
                augmented_masks[idx] = transform_adv_prob(inputs[idx].unsqueeze(0), masks[idx].unsqueeze(0), model, delta)

            # augmented_labels[idx] = 0  # Set the corresponding label to 0

        else:
            # do the augmentation
            masks[idx] = label_augmentation(masks[idx])

            # do the one hot encoding
            one_hot = torch.nn.functional.one_hot(masks[idx].long(), num_classes).permute(0, 3, 1, 2).squeeze(0)


            rand_prob = random.uniform(0.0, 1.0)

            if rand_prob > prob_to_smooth:
                out = one_hot
                # print('no smoothing')
            else:
                smooth_max = random.uniform(0.0, label_smoothing)
                out = nonuniform_label_smooth(one_hot.unsqueeze(0), smooth_max)
                # print('smoothing')

            augmented_masks[idx] = out



    return augmented_masks, augmented_labels
'''


def transform_adversarial_batch(inputs, masks, model, delta, num_classes=3):
    # Get the batch size
    B = masks.shape[0]
    
    # Determine the indices for the half batch to augment
    indices_to_augment = random.sample(range(B), B // 2)

    # print('indices_to_augment:', indices_to_augment)
    
    # Create a copy of masks and labels
    # augmented_masks = masks.clone()
    # augmented_masks = torch.zeros((B, num_classes, masks.shape[2], masks.shape[3])).to(masks.device)
    # augmented_masks = nn.functional.one_hot(masks.long(), num_classes).permute(0, 3, 1, 2).squeeze(0)

    # print('masks:', masks.shape)
    
    augmented_masks = nn.functional.one_hot(masks.long().squeeze(1), num_classes).permute(0, 3, 1, 2).float()

    # print('augmented_masks:', augmented_masks.shape) # (B, num_classes, H, W)
    # Apply the 
    
    # for idx in range(B):
    for idx in indices_to_augment:
        # if transform_type == 'adv':
        # print(inputs[idx].unsqueeze(0).shape, masks[idx].unsqueeze(0).shape)
        augmented_masks[idx] = transform_adv_prob_detach(inputs[idx].unsqueeze(0), masks[idx].unsqueeze(0), model, delta)

    return augmented_masks




if __name__ == '__main__':
    print(os.getcwd())
    print(os.listdir(os.getcwd()))

    path_model = '../../Models/tta_seg/ACDC/unet'
    print(os.listdir(path_model))

    model_dir = os.path.join(path_model, 'best.pt')

    device = 'cuda'

    model = Unet3DSeg(numclasses=3, inshape=(256, 256)).load(model_dir, device)
    model.to(device)
    model.eval()


    # Test the function
    inputs = torch.rand(4, 1, 256, 256)
    masks = torch.rand(4, 1, 256, 256)
    # labels = torch.randint(0, 2, (4, 1))
    labels = torch.ones(4, 1)

    inputs = inputs.to(device)
    masks = masks.to(device)
    labels = labels.to(device)

    delta = 1e-3
    
    # augmented_masks, augmented_labels = perturb_half_batch(inputs, masks, labels, model, delta, 'normal')

    # augmented_masks, augmented_labels = perturb_half_batch_prob(inputs, masks, labels, model, delta, 'adv', num_classes=3)
    # augmented_masks, augmented_labels = perturb_half_batch_prob_patch(inputs, masks, labels, model, delta, 'adv', num_classes=3, number_of_block=8)
    augmented_masks = transform_adversarial_batch(inputs, masks, model, delta, num_classes=3)

    print(augmented_masks.shape)


    # print(augmented_labels)
    # print(augmented_masks[0, 0, 0, 0, 0], augmented_labels[0, 0, 0, 0])