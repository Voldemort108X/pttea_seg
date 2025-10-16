import os
import random
import argparse
import time
import numpy as np
import torch
import glob
from torch.utils.data import DataLoader
from scipy.io import loadmat, savemat
import torch.nn as nn
import time

import copy

from models.segmentation import Unet3DSeg
from models.energy import ShapeEnergyModel
from dataset import MRI_2D_Dataset, GMSCDataset, ChestXrayDataset

# parse the arguments
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--dataset', required=True, help='Name of the dataset')
parser.add_argument('--numclasses', type=int, required=True, help='number of classes in the dataset')
parser.add_argument('--model_seg_dir', required=True,
                    help='model output directory (default: models)')
parser.add_argument('--model_energy_dir', required=True, help='model output directory (default: models)')
parser.add_argument('--batch_size', required=True, type=int, help='batch size for training')
parser.add_argument('--result_dir', required=True, help='result directory')


# optimization parameters
parser.add_argument('--num_iterations', type=int, default=10, help='number of epochs to train (default: 100)')
parser.add_argument('--n_blocks', type=int, default=16, help='number of blocks for the patchwise energy model')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')

                
# seg model parameters
parser.add_argument('--model_type', type=str, default='unet', help='type of model to train (default: unet)')

args = parser.parse_args()

assert args.dataset in [
                        'ACDC',
                        'MnM',
                        'LVQuant',
                        'MyoPS',
                        'gmsc_site1',
                        'gmsc_site2',
                        'gmsc_site3',
                        'gmsc_site4',
                        'chn',
                        'mcu',
                        'jsrt'
                        ]

assert args.model_type in ['unet']

print(os.listdir('../../'))


test_files = glob.glob(os.path.join('../../Dataset', args.dataset, 'test/*.mat'))

assert len(test_files) > 0, 'Could not find any testing data.'

if args.dataset == 'ACDC' or args.dataset == 'MnM' or args.dataset == 'LVQuant' or args.dataset == 'MyoPS':
    # test_set = MRI_2D_Dataset_Classifier(os.path.join('../../Dataset/', args.dataset, 'test'), if_test=True)
    test_set = MRI_2D_Dataset(os.path.join('../../Dataset/', args.dataset, 'test'), if_test=True)
if args.dataset == 'gmsc_site1' or args.dataset == 'gmsc_site2' or args.dataset == 'gmsc_site3' or args.dataset == 'gmsc_site4' or args.dataset == 'gmsc_site2_all' or args.dataset == 'gmsc_site2_split1':
    test_set = GMSCDataset(os.path.join('../../Dataset/', args.dataset, 'test'), if_test=True)
if args.dataset == 'chn' or args.dataset == 'mcu' or args.dataset == 'jsrt':
    test_set = ChestXrayDataset(os.path.join('../../Dataset/', args.dataset, 'test'), if_test=True)

test_data_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

print('Number of files for testing:', len(test_data_loader.dataset))

# inshape
inshape = np.squeeze(test_data_loader.dataset[0][0]).shape
infeats = test_data_loader.dataset[0][0].shape[0]
print('input shape', inshape)
if infeats == 3:
    # for the rgb image, it is still 2D
    inshape = (inshape[1], inshape[2])
ndim = len(inshape)
print('ndim:', ndim)


# prepare model folder
model_energy_dir = args.model_energy_dir

# prepare result folder
result_dir = args.result_dir
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# device handling
device = 'cuda'

# enabling cudnn determinism appears to speed up
torch.backends.cudnn.deterministic = True

if args.model_type == 'unet':
    model_seg = Unet3DSeg(numclasses=args.numclasses, inshape=inshape, src_feats=infeats).load(args.model_seg_dir, device)

model_seg.to(device)
model_seg.eval()

model_seg_copy = copy.deepcopy(model_seg)

# define the optimizer and the rest of the adaptation parameters
optimizer = torch.optim.Adam([p for p in model_seg.parameters() if p.requires_grad], lr=args.lr)
criterion = nn.BCEWithLogitsLoss()
sigmoid_fn = nn.Sigmoid()
softmax_fn = nn.Softmax(dim=1)

# load the energy model
model = ShapeEnergyModel(inshape=inshape, num_classes=args.numclasses, patch_size=args.n_blocks).load(model_energy_dir, device)
model.to(device)
model.eval()

print(model)

def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model

# configure the model
model_seg = configure_model(model_seg)

# check the updated parameters
for name, param in model_seg.named_parameters():
    if param.requires_grad:
        print(name, param.requires_grad)


test_set = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)


start_time = time.time()
for batch in test_data_loader:

    # ----------------- reset the model after each batch -----------------
    model_seg = copy.deepcopy(model_seg_copy)
    model_seg = configure_model(model_seg)
    model_seg.to(device)
    model_seg.eval()

    optimizer = torch.optim.Adam([p for p in model_seg.parameters() if p.requires_grad], lr=args.lr)
    # --------------------------------------------------------------------

    # inputs_test, masks_test, _, file_name = batch
    inputs_test, masks_test, file_name = batch

    inputs_test = inputs_test.to(device)
    masks_test = masks_test.to(device)
    # labels_test = labels_test.to(device)

    inputs_test = inputs_test.float()
    masks_test = masks_test.float()

    n_sample = inputs_test.size(0)

    # define the target label
    target_label = torch.zeros((n_sample, 1, args.n_blocks, args.n_blocks)).to(device)
    target = target_label.view(n_sample, 1, args.n_blocks, args.n_blocks).float()


    for iteration in range(args.num_iterations):

        optimizer.zero_grad()

        y_preds = model_seg(inputs_test)
        y_preds_prob = softmax_fn(y_preds)
        y_preds_prob = y_preds_prob.float()


        # energy model forward pass
        output = model(y_preds_prob)

        # save the initial score and prediction
        if iteration == 0:
            output_init = output.detach().clone()
            y_preds_prob_init = y_preds_prob.detach().clone()
            print('y_preds_prob_init:', y_preds_prob_init.shape)

        # compute the loss to do gradient ascent
        loss = - criterion(output, target)

        # backpropagate
        loss.backward()

        # update the model
        optimizer.step()

        print('Iteration: {}/{}, Loss: {}'.format(iteration, args.num_iterations, loss.item()))

    # mask the y_preds_prob with final score
    output_score = sigmoid_fn(model(y_preds_prob))
    upsampled_score = nn.functional.interpolate(output_score, size=(inshape[0], inshape[0]), mode='bilinear', align_corners=False)
    upsampled_mask = torch.where(upsampled_score < 0.5, 0, 1)
    y_preds_prob_mask = y_preds_prob * upsampled_mask

    output_score_init = sigmoid_fn(output_init)
    upsampled_score_init = nn.functional.interpolate(output_score_init, size=(inshape[0], inshape[0]), mode='bilinear', align_corners=False)
    upsampled_mask_init = torch.where(upsampled_score_init < 0.5, 0, 1)
    y_preds_prob_mask_init = y_preds_prob_init * upsampled_mask_init


    # save the results
    for idx in range(n_sample):
        save_file = {'im': inputs_test.cpu().detach().numpy()[idx,0], 
                     'label': masks_test.cpu().detach().numpy()[idx, 0].astype(np.uint8), 
                     'pred_prob': y_preds_prob.cpu().detach().numpy()[idx],
                     'pred_init': y_preds_prob_init.argmax(dim=1).cpu().detach().numpy()[idx].astype(np.uint8),
                     'pred': y_preds_prob.argmax(dim=1).cpu().detach().numpy()[idx].astype(np.uint8), 
                     'pred_mask': y_preds_prob_mask.argmax(dim=1).cpu().detach().numpy()[idx].astype(np.uint8),
                     'pred_mask_init': y_preds_prob_mask_init.argmax(dim=1).cpu().detach().numpy()[idx].astype(np.uint8),
                     'energy_init': sigmoid_fn(output_init).cpu().numpy()[idx,0],
                     'energy_final': sigmoid_fn(output_score).detach().cpu().numpy()[idx,0]
                     }
        savemat(os.path.join(result_dir, file_name[idx]), save_file)


end_time = time.time()
print('Total time taken:', end_time - start_time)