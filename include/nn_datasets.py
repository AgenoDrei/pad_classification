import os
from collections import Counter
from os.path import join
import random
import cv2
import include.nn_utils
import numpy as np
import pandas as pd
import torch
import include.utils as utl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from include.nn_utils import get_video_desc
from torch.utils.data import Dataset


class RetinaDataset(Dataset):
    def __init__(self, csv_file, root_dir, file_type='.png', class_iloc=1, balance_ratio=1.0, augmentations=None,
                 use_prefix=False):
        """
        Retina Dataset for normal single frame data samples
        :param csv_file: path to csv file with labels
        :param root_dir: path to folder with sample images
        :param file_type: file ending of images (e.g '.jpg')
        :param balance_ratio: adjust sample weight in case of unbalanced classes
        :param transform: pytorch data augmentation
        :param augmentations: albumentation data augmentation
        :param use_prefix: data folder contains subfolders for classes (pos / neg)
        :param boost_frames: boost frames if a third weak prediciton column is available
        """
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.file_type = file_type
        self.augs = augmentations
        self.ratio = balance_ratio
        self.use_prefix = use_prefix
        self.class_iloc = class_iloc
        self.class_freq = {0: sum([1 for v in self.labels_df.iloc[:, class_iloc] if v == 0]), 1: sum([1 for v in self.labels_df.iloc[:, class_iloc] if v >= 1])}
        print(self.class_freq)

    def __len__(self):
        return len(self.labels_df)

    def get_weight(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        severity = 1 if self.labels_df.iloc[idx, self.class_iloc] > 0 else 0
        weight = 1.0 / self.class_freq[severity]
        return weight

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        severity = self.labels_df.iloc[idx, self.class_iloc]
        severity = 1 if severity > 0 else 0

        if self.use_prefix:
            prefix = 'pos' if severity > 0 else 'neg'
        else:
            prefix = ''
        img_name = os.path.join(self.root_dir, prefix, self.labels_df.iloc[idx, 0] + self.file_type)
        img = cv2.imread(img_name)
        assert img is not None, f'Image {img_name} has to exist'

        sample = {'orig_img': img, 'image': img, 'label': severity, 'max_fs': self.labels_df.iloc[idx, self.class_iloc], 'name': os.path.basename(img_name)}
        if self.augs:
            sample['image'] = self.augs(image=img)['image']
        return sample


class SegmentsDataset(Dataset):



########################## Dataset Helper Methods #########################
def get_validation_pipeline(image_size, crop_size):
    return A.Compose([
        A.Resize(image_size, image_size, always_apply=True, p=1.0),
        A.CenterCrop(crop_size, crop_size, always_apply=True, p=1.0),
        A.Normalize(always_apply=True, p=1.0),
        ToTensorV2(always_apply=True, p=1.0)
    ], p=1.0)


def get_training_pipeline(image_size, crop_size, strength=0):
    pipe = A.Compose([
        A.Resize(image_size, image_size, always_apply=True, p=1.0),
        A.RandomCrop(crop_size, crop_size, always_apply=True, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.3),
        # border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
        A.OneOf([A.GaussNoise(p=0.5), A.ISONoise(p=0.5), A.IAAAdditiveGaussianNoise(p=0.25), A.MultiplicativeNoise(p=0.25)], p=0.3),
        A.OneOf([A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5), A.GridDistortion(p=0.5)], p=0.3),
        A.OneOf([A.HueSaturationValue(p=0.5), A.ToGray(p=0.5), A.RGBShift(p=0.5)], p=0.3),
        A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2), A.RandomGamma()], p=0.3),
        A.Normalize(always_apply=True, p=1.0),
        ToTensorV2(always_apply=True, p=1.0)
    ], p=1.0)
    return pipe
