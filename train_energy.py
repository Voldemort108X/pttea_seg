import os
import random
import argparse
import time
import numpy as np
import torch
import glob
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import wandb

from models.segmentation import Unet3DSeg
from models.energy import ShapeEnergyModel
# from networks_patch_old import ShapeClassifierLogits_patch

from scheduler import WarmUpCosineDecayScheduler

from dataset import *
from perturbation import transform_adversarial_batch

from augmentation import SpatialAugmentation

from utils import convert_to_onehot, apply_random_augmentation_onehot, create_labels, create_labels_iou

parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--dataset', required=True, help='Name of the dataset')
parser.add_argument('--model-dir', required=True,
                    help='model output directory (default: models)')

# training parameters
parser.add_argument('--numclasses', type=int, required=True, help='number of classes in the dataset')
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=4, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=150,
                    help='number of training epochs (default: 150)')
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--save_freq', type=int, default=20, 
                    help='save frequency')
parser.add_argument('--patch_size', type=int, default=16, help='patch size (default: 16)')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')

# loss hyperparameters
parser.add_argument('--image-loss', required=True,
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
# adv model parameters
parser.add_argument('--transform_type', required=True, help='type of transform used')
parser.add_argument('--load-seg-model', help='optional model file to initialize with')

# augmentation function
parser.add_argument('--augmentation', required=True, help='type of augmentation used')
parser.add_argument('--aug_prob', type=float, default=0.5, help='probability of augmentation')
parser.add_argument('--max_delta', type=float, default=1e-1, help='maximum delta value for perturbation')

# create label threshold
parser.add_argument('--threshold', type=int, default=50, help='threshold for creating labels')
parser.add_argument('--iou_threshold', type=float, default=0.6, help='threshold for creating labels')
parser.add_argument('--use_iou', action='store_true', help='use iou threshold for creating labels')

# abaltion for no adversarial perturbation
parser.add_argument('--no_adv', action='store_true', help='no adversarial perturbation')

# wandb run name
parser.add_argument('--wandb-name', type=str, required=True, help='name of wandb run')

# model_seg parameters
parser.add_argument('--model_seg_type', required=True, help='model type (default: unet)')

parser.add_argument('--debug', action='store_true', help='debug mode, only run one iteration')

args = parser.parse_args()


assert args.dataset in [
                        'ACDC',
                        'MnM',
                        'LVQuant',
                        'MyoPS',
                        'CAMUS',
                        'gmsc_site1',
                        'gmsc_site2',
                        'gmsc_site3',
                        'gmsc_site4',
                        'chn',
                        'mcu',
                        'jsrt'
                        ]

assert args.image_loss in ['bce']
# assert args.label_name in ['endo_seg', 'epi_seg']
assert args.augmentation in ['spatial_aug']
assert args.model_seg_type in ['unet']

print(os.listdir('../../'))
train_files = glob.glob(os.path.join('../../Dataset/', args.dataset, 'train/*.mat')) 
valid_files = glob.glob(os.path.join('../../Dataset', args.dataset, 'val/*.mat'))
assert len(train_files) > 0, 'Could not find any training data.'
assert len(valid_files) > 0, 'Could not find any validation data.'

if args.dataset == 'ACDC' or args.dataset == 'MnM' or args.dataset == 'LVQuant' or args.dataset == 'MyoPS':
    train_set = MRI_2D_Dataset(os.path.join('../../Dataset/', args.dataset, 'train'))
    valid_set = MRI_2D_Dataset(os.path.join('../../Dataset/', args.dataset, 'val'))

elif args.dataset == 'gmsc_site1' or args.dataset == 'gmsc_site2' or args.dataset == 'gmsc_site3' or args.dataset == 'gmsc_site4':
    train_set = GMSCDataset(os.path.join('../../Dataset/', args.dataset, 'train'))
    valid_set = GMSCDataset(os.path.join('../../Dataset/', args.dataset, 'val'))

elif args.dataset == 'chn' or args.dataset == 'mcu' or args.dataset == 'jsrt':
    train_set = ChestXrayDataset(os.path.join('../../Dataset/', args.dataset, 'train'))
    valid_set = ChestXrayDataset(os.path.join('../../Dataset/', args.dataset, 'val'))

train_data_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
valid_data_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)

print('Number of files for training:', len(train_data_loader.dataset))
print('Number of files for validation:', len(valid_data_loader.dataset))

# inshape
inshape = np.squeeze(train_data_loader.dataset[0][0]).shape
infeats = train_data_loader.dataset[0][0].shape[0]
print('input shape', inshape)
if infeats == 3:
    # for the rgb image, it is still 2D
    inshape = (inshape[1], inshape[2])
ndim = len(inshape)
print('ndim', ndim)

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
# device = 'cuda' # cuda
if args.debug:
    device = 'cpu'
else:
    device = 'cuda'
print('running on device', device)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert np.mod(args.batch_size, nb_gpus) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = True

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]


if args.load_model:
    # load initial model (if specified)
    # model = ShapeClassifierLogits.load(args.load_model, device)
    model = ShapeEnergyModel.load(args.load_model, device)
    # model = torch.load(args.load_model)
    model.to(device)
else:
    # otherwise configure new model
    model = ShapeEnergyModel(
        inshape = inshape,
        num_classes=args.numclasses,
        patch_size=args.patch_size
    )


# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
total_steps = args.epochs * len(train_data_loader)  # Assuming one step per batch
print('total steps', total_steps)
scheduler = WarmUpCosineDecayScheduler(optimizer, warmup_steps=1000, total_steps=total_steps)

# prepare image loss
if args.image_loss == 'bce':
    loss_func = nn.BCEWithLogitsLoss()
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# load segmentation model
checkpoints = os.listdir(args.load_seg_model)
assert len(checkpoints) > 0, 'Could not find any checkpoints in the model directory.'
print('Number of checkpoints found:', len(checkpoints))

# only keep the last 5 checkpoints
# checkpoints = checkpoints[-5:]
# remove the 0000.pt checkpoint
checkpoints = [c for c in checkpoints if c != '0000.pt']
print('Number of checkpoints after filtering:', len(checkpoints))

if args.model_seg_type == 'unet':
    delta_list = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    # Filter delta_list to keep only values <= max_delta
    delta_list = [d for d in delta_list if d <= args.max_delta]

print('delta_list', delta_list)

# wandb tracking experiments
run = wandb.init(
    # set the wandb project where this run will be logged
    project="YOUR_PROJECT_NAME",

    dir='../../Wandb',
    
    # track hyperparameters and run metadata
    config={
    # "numclasses": args.numclasses,
    "learning_rate": args.lr,
    "architecture": "UNet",
    "dataset": args.dataset,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "enc_nf": enc_nf,
    "dec_nf": dec_nf,
    "image-loss": args.image_loss,
    },

    # have a run name
    name = args.wandb_name
)

if args.augmentation == 'spatial_aug':
    if args.model_seg_type == 'unet':
        aug_func = SpatialAugmentation(device=device, p=args.aug_prob, noise_prob=0.09, patch_size=(16, 16), num_classes=args.numclasses)

# training loops
best_val_loss = np.inf

epoch_time = []
for epoch in range(args.initial_epoch, args.epochs):
    epoch_start_time = time.time()

    # save model checkpoint
    if epoch % args.save_freq == 0:
        model.save(os.path.join(model_dir, '%04d.pt' % epoch))

    # training loop
    model.to(device)
    model.train()

    epoch_train_loss = []
    for inputs_train, masks_train in tqdm(train_data_loader):
        # print('inputs shape', inputs_train.shape)
        inputs_train = inputs_train.to(device)
        masks_train = masks_train.to(device)

        # convert tensor type to float
        inputs_train = inputs_train.float()
        masks_train = masks_train.float()

        # load the segmentation model
        rand_checkpoint = random.choice(checkpoints)

        if args.model_seg_type == 'unet':
            model_seg = Unet3DSeg(numclasses=args.numclasses, inshape=inshape, src_feats=infeats).load(os.path.join(args.load_seg_model, rand_checkpoint), device)
        
        model_seg.to(device)
        model_seg.eval()

        rand_delta = random.choice(delta_list)
        # rand_delta = torch.normal(mean=delta_mean, std=delta_std)
        # rand_delta = np.random.normal(loc=delta_mean, scale=delta_std)
        rand_delta = np.abs(rand_delta)

        if args.no_adv:
            masks_train_perturb = model_seg(masks_train)
        else:
            masks_train_perturb = transform_adversarial_batch(inputs_train, masks_train, model_seg, rand_delta, num_classes=args.numclasses) # B x C x H x W, half augmented half real

        masks_train_onehot = convert_to_onehot(masks_train, args.numclasses)

        masks_train_onehot = masks_train_onehot.float()
        masks_train_perturb = masks_train_perturb.float()

        masks_train_onehot, masks_train_perturb = apply_random_augmentation_onehot(masks_train_onehot, masks_train_perturb, aug_func, numclasses=args.numclasses)

        if args.use_iou:
            labels_train = create_labels_iou(masks_train_onehot.float(), masks_train_perturb.float(), n_blocks=args.patch_size, iou_threshold=args.iou_threshold)
        else:
            labels_train = create_labels(masks_train_onehot.float(), masks_train_perturb.float(), n_blocks=args.patch_size, threshold=args.threshold)
      
        y_pred_train = model(masks_train_perturb.float())

        loss_train = loss_func(y_pred_train.float(), labels_train.float()) # (input, target)

        epoch_train_loss.append(loss_train.item())
       
        loss_train.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if args.debug:
            break
    
    # validation loop
    model.eval()
    # with torch.no_grad(): # cannot remove since the adversarial perturbation needs gradient update 
    if True:
        epoch_val_loss = []
        # for inputs_val, masks_val, _ in tqdm(valid_data_loader):
        for inputs_val, masks_val in tqdm(valid_data_loader):
            inputs_val = inputs_val.to(device)
            masks_val = masks_val.to(device)
            # labels_val = labels_val.to(device) # labels_val shape N x H x W x D

            # convert tensor type to float
            inputs_val = inputs_val.float()
            masks_val = masks_val.float()
            # labels_val = labels_val.float()

            # load the segmentation model
            rand_checkpoint = random.choice(checkpoints)
            if args.model_seg_type == 'unet':
                model_seg = Unet3DSeg(numclasses=args.numclasses, inshape=inshape).load(os.path.join(args.load_seg_model, rand_checkpoint), device)
        
            model_seg.to(device)
            model_seg.eval()

            rand_delta = random.choice(delta_list)

            if args.no_adv:
                masks_val_perturb = model_seg(masks_val)
            else:
                masks_val_perturb = transform_adversarial_batch(inputs_val, masks_val, model_seg, rand_delta, num_classes=args.numclasses)

            masks_val_onehot = convert_to_onehot(masks_val, args.numclasses)

            masks_val_onehot = masks_val_onehot.float()
            masks_val_perturb = masks_val_perturb.float()

            masks_val_onehot, masks_val_perturb = apply_random_augmentation_onehot(masks_val_onehot, masks_val_perturb, aug_func, numclasses=args.numclasses)

            if args.use_iou:
                labels_val = create_labels_iou(masks_val_onehot.float(), masks_val_perturb.float(), n_blocks=args.patch_size, iou_threshold=args.iou_threshold)
            else:
                labels_val = create_labels(masks_val_onehot.float(), masks_val_perturb.float(), n_blocks=args.patch_size, threshold=args.threshold)
            

            y_pred_val = model(masks_val_perturb.float())

            loss_val = loss_func(y_pred_val.float(), labels_val.float()) # (input, target)
            
            epoch_val_loss.append(loss_val.item())

            if args.debug:
                break

        # save best model
        if np.mean(epoch_val_loss) < best_val_loss:
            best_val_loss = np.mean(epoch_val_loss)
            model.save(os.path.join(model_dir, 'best.pt'))

    # track epoch time
    epoch_time.append(time.time() - epoch_start_time)

     # track gpu memory
    memory_allocated = torch.cuda.memory_allocated(torch.cuda.current_device())
    memory_reserved = torch.cuda.memory_reserved(torch.cuda.current_device())

    epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
    train_info = 'train loss: %.4f' % np.mean(epoch_train_loss)
    val_info = 'val loss: %.4f' % np.mean(epoch_val_loss)
    time_info = '%.4f sec/epoch' % np.mean(epoch_time)
    memory_info = 'GPU Memory Allocated: %.2f MB, Reserved: %.2f MB' % (memory_allocated / (1024**2), memory_reserved / (1024**2))

    print(' - '.join((epoch_info, time_info, train_info, val_info, memory_info)), flush=True)

    # extract one sample idx in the batch such that is is augmented (label=0)

    train_masks = wandb.Image(masks_train_onehot.argmax(dim=1).cpu().detach().numpy()[0]/(args.numclasses-1))
    train_masks_perturb = wandb.Image(masks_train_perturb.argmax(dim=1).cpu().detach().numpy()[0]/(args.numclasses-1))
    train_labels  = wandb.Image(labels_train.cpu().detach().numpy()[0, 0])
    train_pred = wandb.Image(y_pred_train.cpu().detach().numpy()[0, 0])

    val_masks = wandb.Image(masks_val_onehot.argmax(dim=1).cpu().detach().numpy()[0]/(args.numclasses-1))
    val_masks_perturb = wandb.Image(masks_val_perturb.argmax(dim=1).cpu().detach().numpy()[0]/(args.numclasses-1))
    val_labels  = wandb.Image(labels_val.cpu().detach().numpy()[0, 0])
    val_pred = wandb.Image(y_pred_val.cpu().detach().numpy()[0, 0])

    wandb.log({
        "train_loss": np.mean(epoch_train_loss),
        "val_loss": np.mean(epoch_val_loss),

        "train_masks": train_masks,
        "train_masks_perturb": train_masks_perturb,
        "train_labels": train_labels,
        "train_pred": train_pred,

        "val_masks": val_masks,
        "val_masks_perturb": val_masks_perturb,
        "val_labels": val_labels,
        "val_pred": val_pred,
    })

model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))