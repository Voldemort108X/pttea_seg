import torch.utils.data as data
import numpy as np
import torch
import os
from scipy.io import loadmat, savemat

def im_normalize(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img


class MRI_2D_Dataset_Classifier(data.Dataset):
    def __init__(self, data_path, if_test=False):
        super(MRI_2D_Dataset_Classifier, self).__init__()
        self.data_path = data_path
        self.file_names = os.listdir(data_path)
        self.if_test = if_test
        super().__init__()

    def __getitem__(self, index):

        if self.if_test:
            input, mask, label, file_name = load_data_mri_2d_classifier(self.data_path, self.file_names[index], self.if_test)

            return input, mask, label, file_name
        else:
            input, mask, label = load_data_mri_2d_classifier(self.data_path, self.file_names[index], self.if_test)
            return input, mask, label

    def __len__(self):
        return len(self.file_names)

def load_data_mri_2d_classifier(data_path, file_name, if_test=False):
    # assert label_name in ['endo_seg', 'epi_seg']
    file = loadmat(os.path.join(data_path, file_name))

    input = file['im']
    input = im_normalize(input)
    input = input[np.newaxis, ...]

    mask = np.zeros_like(file['myo_seg']) # label shape N x H x W x D
    mask[file['myo_seg']>0.5] = 2 # label myocardium as 2
    mask[file['endo_seg']>0.5] = 1 # label endocardium as 1 and the rest 2 becomes myocardium
    mask = mask[np.newaxis, ...]
   
    label = 1
    label = np.array(label)
    label = label[np.newaxis, ...]


    if if_test:
        return input, mask, label, file_name
    else:
        return input, mask, label


#################################################
# Datasets from unet/dataset.py EXCEPT adding a new dimension for label 
#################################################


class MRI_2D_Dataset(data.Dataset):
    def __init__(self, data_path, if_test=False):
        super(MRI_2D_Dataset, self).__init__()
        self.data_path = data_path
        self.file_names = os.listdir(data_path)
        self.if_test = if_test
        super().__init__()

    def __getitem__(self, index):

        if self.if_test:
            input, label, file_name = load_data_mri_2d(self.data_path, self.file_names[index], self.if_test)
            return input, label, file_name
        else:
            input, label = load_data_mri_2d(self.data_path, self.file_names[index], self.if_test)
            return input, label

    def __len__(self):
        return len(self.file_names)

def load_data_mri_2d(data_path, file_name, if_test=False):
    # assert label_name in ['endo_seg', 'epi_seg']
    file = loadmat(os.path.join(data_path, file_name))

    input = file['im']
    input = im_normalize(input)

    # label = file[label_name]
    # label_endo = file['endo_seg']
    # label_myo = file['epi_seg'] - file['endo_seg']

    # label_endo ==1 and label_myo == 2
    label = np.zeros_like(file['myo_seg']) # label shape N x H x W x D
    label[file['myo_seg']>0.5] = 2 # label myocardium as 2
    label[file['endo_seg']>0.5] = 1 # label endocardium as 1 and the rest 2 becomes myocardium
   
    input = input[np.newaxis, ...]
    label = label[np.newaxis, ...] # required for train_patch

    if if_test:
        return input, label, file_name
    else:
        return input, label


class GMSCDataset(data.Dataset):
    def __init__(self, data_path, if_test=False):
        super(GMSCDataset, self).__init__()
        self.data_path = data_path
        self.file_names = os.listdir(data_path)
        self.if_test = if_test
        super().__init__()

    def __getitem__(self, index):

        if self.if_test:
            input, label, file_name = load_data_gmsc(self.data_path, self.file_names[index], self.if_test)
            return input, label, file_name
        else:
            input, label = load_data_gmsc(self.data_path, self.file_names[index])
            return input, label

    def __len__(self):
        return len(self.file_names)


def load_data_gmsc(data_path, file_name, if_test=False):
    file = loadmat(os.path.join(data_path, file_name))

    input = file['im']
    input = im_normalize(input)

    gm_mask = file['gm_mask']
    wm_mask = file['wm_mask']

    # label = file['seg']
    # add gm and wm mask but make them stricly 1s and 0s
    label = (gm_mask + wm_mask) > 0

    label = label[np.newaxis, ...] # follow the convention in the train.py

    # check the outlier
    if np.max(label) > 1:
        print('Outlier in', file_name)

    input = input[np.newaxis, ...]

    if if_test:
        return input, label, file_name
    else:  
        return input, label


class ChestXrayDataset(data.Dataset):
    def __init__(self, data_path, if_test=False):
        super(ChestXrayDataset, self).__init__()
        self.data_path = data_path
        self.file_names = os.listdir(data_path)
        self.if_test = if_test
        super().__init__()

    def __getitem__(self, index):

        if self.if_test:
            input, label, file_name = load_data_chestxray(self.data_path, self.file_names[index], self.if_test)
            return input, label, file_name
        else:
            input, label = load_data_chestxray(self.data_path, self.file_names[index])
            return input, label

    def __len__(self):
        return len(self.file_names)


def load_data_chestxray(data_path, file_name, if_test=False):
    file = loadmat(os.path.join(data_path, file_name))

    input = file['im']
    input = im_normalize(input)

    mask = file['mask']

    # label = file['seg']
    # add gm and wm mask but make them stricly 1s and 0s
    label = mask > 0

    label = label[np.newaxis, ...] #follow the convention in the train.py

    input = input[np.newaxis, ...]

    if if_test:
        return input, label, file_name
    else:  
        return input, label