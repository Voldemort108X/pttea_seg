import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


from .modelio import LoadableModel, store_config_args

class ShapeEnergyModel(LoadableModel):
    @store_config_args
    def __init__(self, inshape, num_classes, patch_size=16):
        super(ShapeEnergyModel, self).__init__()
        
        # Determine the number of dimensions in the input
        self.ndims = len(inshape)  
        self.num_classes = num_classes

        Conv = getattr(nn, 'Conv%dd' % self.ndims)
        Norm = getattr(nn, 'BatchNorm%dd' % self.ndims)
        # MaxPool = getattr(nn, 'MaxPool%dd' % self.ndims)

        n_downsample = inshape[0] // patch_size

        # print('n_downsample:', n_downsample)


        # Define the network layers
        if n_downsample == 16: # patch size = 16
            self.features = nn.Sequential(
                Conv(num_classes, 64, kernel_size=5, stride=2, padding=2),  # Changed to 5x5, stride 1
                nn.LeakyReLU(0.2, inplace=True),
                Conv(64, 128, kernel_size=5, stride=2, padding=2),
                Norm(128),
                # MaxPool(kernel_size=2, stride=2),  # Added max pooling
                nn.LeakyReLU(0.2, inplace=True),
                Conv(128, 256, kernel_size=5, stride=2, padding=2),
                Norm(256),
                # MaxPool(kernel_size=2, stride=2),  # Added max pooling
                nn.LeakyReLU(0.2, inplace=True),
                Conv(256, 512, kernel_size=5, stride=2, padding=2),
                Norm(512),
                # MaxPool(kernel_size=2, stride=2),  # Added max pooling
                nn.LeakyReLU(0.2, inplace=True),
            )

            self.classifier_head = Conv(512, 1, kernel_size=3, stride=1, padding=1)
        
        elif n_downsample == 36: # patch size = 4 for input 144x144
            self.features = nn.Sequential(
                Conv(num_classes, 64, kernel_size=5, stride=2, padding=2),  # Changed to 5x5, stride 1
                nn.LeakyReLU(0.2, inplace=True),
                Conv(64, 128, kernel_size=5, stride=2, padding=2),
                Norm(128),
                nn.LeakyReLU(0.2, inplace=True),
                Conv(128, 256, kernel_size=5, stride=2, padding=2),
                Norm(256),
                nn.LeakyReLU(0.2, inplace=True),
                Conv(256, 512, kernel_size=5, stride=2, padding=2),
                Norm(512),
                nn.LeakyReLU(0.2, inplace=True),
                Conv(512, 512, kernel_size=3, stride=2, padding=1),
                Norm(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AdaptiveAvgPool2d((4, 4))
            )

            self.classifier_head = Conv(512, 1, kernel_size=3, stride=1, padding=1)
        
        elif n_downsample == 8: # patch size = 32
            self.features = nn.Sequential(
                Conv(num_classes, 64, kernel_size=5, stride=2, padding=2),  # Changed to 5x5, stride 1
                nn.LeakyReLU(0.2, inplace=True),
                Conv(64, 128, kernel_size=5, stride=2, padding=2),
                Norm(128),
                nn.LeakyReLU(0.2, inplace=True),
                Conv(128, 256, kernel_size=5, stride=2, padding=2),
                Norm(256),
                nn.LeakyReLU(0.2, inplace=True),
            )

            self.classifier_head = Conv(256, 1, kernel_size=3, stride=1, padding=1)
        
        elif n_downsample == 4: # patch size = 64
            self.features = nn.Sequential(
                Conv(num_classes, 64, kernel_size=5, stride=2, padding=2),  # Changed to 5x5, stride 1
                nn.LeakyReLU(0.2, inplace=True),
                Conv(64, 128, kernel_size=5, stride=2, padding=2),
                Norm(128),
                nn.LeakyReLU(0.2, inplace=True),
            )

            self.classifier_head = Conv(128, 1, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier_head(x)
        return x