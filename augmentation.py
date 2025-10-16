import torch
from typing import Optional
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import random
import torch.nn.functional as F

try:
    from .utils import create_labels
except ImportError:  # allow running without package context
    from utils import create_labels

from monai.transforms import (
    Compose,
    RandFlipd,
    RandAffined,
    ToDeviced,
    RandCoarseDropoutd
)

class SpatialAugmentation: # fix for the number of classes
    def __init__(self, device, num_classes, p=0.5, noise_prob=0.05, patch_size=(16, 16)):
        self.device = device
        self.num_classes = num_classes
        self.p = p
        self.noise_prob = noise_prob  # Probability for salt and pepper noise
        self.patch_size = patch_size  # Size of the random patch for noise
        self.spatial_transforms = Compose([
            ToDeviced(keys=["mask", "mask_adv"], device=self.device),
            RandFlipd(keys=["mask", "mask_adv"], spatial_axis=[1], prob=p),
            RandAffined(
                keys=["mask", "mask_adv"],
                rotate_range=[10, 10],
                scale_range=[0.1, 0.1],
                padding_mode="zeros",
                prob=p,
                device=self.device
            )
        ])

        self.spatial_transforms_mask = Compose([
            ToDeviced(keys=["mask"], device=self.device),
            RandCoarseDropoutd(
                keys=["mask"],
                holes=300,
                spatial_size=(15, 15),
                max_holes=400,
                fill_value=0,
                prob=1,
            ),
            RandAffined(
                keys=["mask"],
                rotate_range=[10, 10],
                scale_range=[0.1, 0.1],
                translate_range=[100, 100],
                padding_mode="zeros",
                prob=1,
                device=self.device
            )
        ])
    
    def add_masks(self, mask1, mask2):
        mask1 = mask1.squeeze()
        mask2 = mask2.squeeze()
        mask_out = mask1.clone()
        mask_out[mask2 != 0] = mask2[mask2 != 0]
        return mask_out.unsqueeze(0)

    def apply_salt_and_pepper_noise_to_patch(self, mask, expansion_size=4):
        # Get mask dimensions
        H, W = mask.shape[-2:]
        ph, pw = self.patch_size

        # Randomly select the top-left corner of the patch
        x = torch.randint(0, H - ph + 1, (1,)).item()
        y = torch.randint(0, W - pw + 1, (1,)).item()

        # Create the patch in mask_adv where salt-and-pepper noise will be applied
        patch = mask[..., x:x+ph, y:y+pw].clone()
        mask_patch = mask[..., x:x+ph, y:y+pw]  # Original mask patch

        # Apply noise only to areas where mask is zero
        noise = torch.rand_like(patch)
        patch[(noise < self.noise_prob / 2) & (mask_patch == 0)] = 0  # Pepper (set to background)
        
        # Dynamically adjust noise values based on the number of classes
        for class_idx in range(1, self.num_classes):
            noise = torch.rand_like(patch)
            patch[(noise < self.noise_prob / 2) & (mask_patch == 0)] = class_idx  # Salt for class class_idx

        # Expand noise to neighboring pixels
        kernel_size = (expansion_size, expansion_size)
        expanded_patch = F.max_pool2d(patch.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=1, padding=expansion_size // 2)
        expanded_patch = expanded_patch.squeeze()  # Remove extra dimensions

        # Ensure the expanded patch has the same size as the original
        expanded_patch = expanded_patch[:ph, :pw]  # Crop if needed to match original patch size

        # Insert the modified patch back into the mask_adv
        mask[..., x:x+ph, y:y+pw] = expanded_patch
        
        return mask

    def apply_spatial_transforms(self, mask, mask_adv):
        B, H, W = mask.shape
        augmented_mask = torch.empty_like(mask)
        augmented_mask_adv = torch.empty_like(mask_adv)
        
        mask = mask.unsqueeze(1)
        mask_adv = mask_adv.unsqueeze(1)
        
        for i in range(B):
            data = {"mask": mask[i], "mask_adv": mask_adv[i]}
            transformed = self.spatial_transforms(data)
            augmented_mask[i] = transformed["mask"]
            augmented_mask_adv[i] = transformed["mask_adv"]

            cropped_mask = self.spatial_transforms_mask({"mask": mask[i]})["mask"]

            rand_number = torch.rand(1)
            if rand_number < self.p:
                augmented_mask_adv[i] = self.add_masks(cropped_mask.float(), augmented_mask_adv[i].float())

            # Apply salt and pepper noise to a random patch in the mask_adv
            augmented_mask_adv[i] = self.apply_salt_and_pepper_noise_to_patch(augmented_mask_adv[i].float())

        return augmented_mask, augmented_mask_adv
    
    def __call__(self, mask, mask_adv):
        mask, mask_adv = self.apply_spatial_transforms(mask, mask_adv)
        return mask.unsqueeze(1), mask_adv.unsqueeze(1)



def nonuniform_label_smooth(array, smooth_max):

    B, C, H, W = array.shape

    softmax_layer = nn.Softmax(dim=1)

    noise = torch.rand((B, C, H, W), device=array.device) * smooth_max

    # noise.to(array.device)

    # print(noise.device, array.device)

    # print('max and min', torch.max(noise), torch.min(noise))

    array_out = array + noise

    array_out = softmax_layer(array_out)

    return array_out

# def add_mask_aug(mask_org, mask_perturb)



class RandomLabelAugmentation:
    def __init__(self, device, p=0.5):
        self.device = device
        self.p = p
        self.spatial_transforms = Compose([
            ToDeviced(keys=["label"], device=self.device),
            RandFlipd(keys=["label"], spatial_axis=[1], prob=p),  # Flip along width
            RandAffined(
                keys=["label"],
                rotate_range=[10, 10],
                scale_range=[0.1, 0.1],
                translate_range=[40, 40],
                padding_mode="zeros",
                prob=p,
                device=self.device
            ),  # Rotate only along the spatial dimensions
        ])
    
    def apply_spatial_transforms(self, label):
        # device = image.device
        B, H, W = label.shape

        # label shape assumed to be B x H x W

        # print(B, C, H, W)
        augmented_label = torch.empty_like(label)

        label = label.unsqueeze(1)  # Add channel dimension
        # print(image.shape, label.shape)
        
        for i in range(B):
            data = {"label": label[i]}
            transformed = self.spatial_transforms(data)
            augmented_label[i] = transformed["label"]
        
        return augmented_label
    
    def __call__(self, label):
        label = self.apply_spatial_transforms(label)
        return label.unsqueeze(1)


if __name__ == '__main__':
    
    mask_org = torch.ones((4, 3, 256, 256))
    mask_aug = torch.zeros((4, 3, 256, 256))

    label = create_labels(mask_org, mask_aug, n_blocks=8)

    print(torch.max(label), torch.min(label))


 
