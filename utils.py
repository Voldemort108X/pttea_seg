import torch.nn.functional as F
import torch
import torch.nn as nn


def convert_to_one_hot(tensor, ndim, num_classes):
    if ndim == 3:
        # print('target shape', target.shape)
        tensor_onehot = F.one_hot(tensor.long(), num_classes).permute(0, 4, 1, 2, 3)
    elif ndim == 2:
        tensor_onehot = F.one_hot(tensor.long(), num_classes).permute(0, 3, 1, 2)

    return tensor_onehot


def convert_to_onehot(mask, num_classes):
    # assert the mask has shape B x 1 x H x W
    # print('inside convert_to_onehot num_classes:', num_classes)
    # print('mask shape:', mask.shape)
    # print('mask max:', torch.max(mask))

    onehot = nn.functional.one_hot(mask.long().squeeze(1), num_classes).permute(0, 3, 1, 2).float()

    return onehot


def onehot_to_mask(onehot):
    # assert the onehot has shape B x C x H x W

    mask = torch.argmax(onehot, dim=1, keepdim=True)

    return mask

def apply_random_augmentation_onehot(mask1, mask2, aug_func, numclasses):
    # print('numclasses:', numclasses)
    mask1 = onehot_to_mask(mask1)
    mask2 = onehot_to_mask(mask2)

    # print(mask1.shape, mask2.shape)

    mask1, mask2 = aug_func(mask1.squeeze(1), mask2.squeeze(1)) # be careful when batchsize=1

    mask1 = convert_to_onehot(mask1, numclasses)
    mask2 = convert_to_onehot(mask2, numclasses)

    return mask1, mask2


def create_labels(mask_org, mask_aug, n_blocks=16, threshold=50):

    # print('mask_org:', mask_org.shape)
    # print('mask_aug:', mask_aug.shape)

    B, C, H, W = mask_org.shape

    kernel_size_h = H // n_blocks
    kernel_size_w = W // n_blocks

    unfold = nn.Unfold(kernel_size=(kernel_size_h, kernel_size_w), stride=(kernel_size_h, kernel_size_w))
    fold = nn.Fold(output_size=(H, W), kernel_size=(kernel_size_h, kernel_size_w), stride=(kernel_size_h, kernel_size_w))

    mask_org_unfolded = unfold(mask_org)
    mask_aug_unfolded = unfold(mask_aug)

    # maybe use the signed distance (-1, 0, 1) -> 3 class classification
    diff = torch.abs(mask_org_unfolded - mask_aug_unfolded)
    diff = torch.sum(diff, dim=1)  # B x (HW)

    label = torch.where(diff > threshold, 0, 1) 

    return label.view(-1, 1, n_blocks, n_blocks)


def create_labels_list(mask_org, mask_aug, n_blocks_list, threshold=50):

    # print('mask_org:', mask_org.shape)
    # print('mask_aug:', mask_aug.shape)

    B, C, H, W = mask_org.shape

    label_list = []
    for n_blocks in n_blocks_list:

        kernel_size_h = H // n_blocks
        kernel_size_w = W // n_blocks

        unfold = nn.Unfold(kernel_size=(kernel_size_h, kernel_size_w), stride=(kernel_size_h, kernel_size_w))
        fold = nn.Fold(output_size=(H, W), kernel_size=(kernel_size_h, kernel_size_w), stride=(kernel_size_h, kernel_size_w))

        mask_org_unfolded = unfold(mask_org)
        mask_aug_unfolded = unfold(mask_aug)

        # maybe use the signed distance (-1, 0, 1) -> 3 class classification
        diff = torch.abs(mask_org_unfolded - mask_aug_unfolded)
        diff = torch.sum(diff, dim=1)  # B x (HW)

        label = torch.where(diff > threshold, 0, 1) 
    
        label_list.append(label.view(-1, 1, n_blocks, n_blocks))

    return label_list

def onehot_to_oneclass(mask_onehot, class_id):
    # assert mask in shape B x C x H x W
    mask = mask_onehot.argmax(dim=1, keepdim=True)

    mask = torch.where(mask == class_id, 1, 0)

    return mask


def create_labels_iou(mask_org, mask_aug, n_blocks=16, iou_threshold=0.6):
    B, C, H, W = mask_org.shape

    kernel_size_h = H // n_blocks
    kernel_size_w = W // n_blocks

    unfold = nn.Unfold(kernel_size=(kernel_size_h, kernel_size_w), stride=(kernel_size_h, kernel_size_w))
    fold = nn.Fold(output_size=(H, W), kernel_size=(kernel_size_h, kernel_size_w), stride=(kernel_size_h, kernel_size_w))

    mask_org_unfolded = unfold(mask_org)
    mask_aug_unfolded = unfold(mask_aug)

    # Calculate intersection and union
    intersection = torch.sum(mask_org_unfolded * mask_aug_unfolded, dim=1)  # B x (HW)
    union = torch.sum(mask_org_unfolded + mask_aug_unfolded - mask_org_unfolded * mask_aug_unfolded, dim=1)  # B x (HW)
    
    # Compute IoU
    iou = intersection / (union + 1e-6)  # Avoid division by zero

    # Label patches based on IoU threshold
    label = torch.where(iou > iou_threshold, 1, 0)

    return label.view(-1, 1, n_blocks, n_blocks)


# def update_input()

if __name__ == '__main__':
    mask_org = torch.randn(1, 1, 144, 144)
    mask_aug = torch.randn(1, 1, 144, 144)

    labels = create_labels(mask_org, mask_aug)
    print(labels.shape)

    labels_list = create_labels_list(mask_org, mask_aug, [18, 9, 4])

    for label in labels_list:
        print(label.shape)
    # print(create_labels(mask_org, mask_aug))

    # print(create_labels_list(mask_org, mask_aug, [18, 9, 4]))
