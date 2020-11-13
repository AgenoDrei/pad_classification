from os.path import join

import albumentations as A
import cv2
import os
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class RetinaDataset(Dataset):
    def __init__(self, csv_file, root_dir, file_type='.png', class_iloc=1, augmentations=None,
                 use_prefix=False, thresh=1):
        """
        Retina Dataset for normal single frame data samples
        :param csv_file: path to csv file with labels
        :param root_dir: path to folder with sample images
        :param file_type: file ending of images (e.g '.jpg')
        :param transform: pytorch data augmentation
        :param augmentations: albumentation data augmentation
        :param use_prefix: data folder contains subfolders for classes (pos / neg)
        :param boost_frames: boost frames if a third weak prediciton column is available
        """
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.file_type = file_type
        self.augs = augmentations
        self.use_prefix = use_prefix
        self.class_iloc = class_iloc
        self.class_threshold = thresh

        self.class_freq = {0: sum([1 for v in self.labels_df.iloc[:, class_iloc] if v < self.class_threshold]),
                           1: sum([1 for v in self.labels_df.iloc[:, class_iloc] if v >= self.class_threshold])}
        self.class_mapping = {0: 0, 1: 1}
        if self.class_freq[0] < self.class_freq[1]:
            print('Switched class labels, not enough negatives!')
            self.class_mapping = {0: 1, 1: 0}
        # print(self.class_freq)

    def __len__(self):
        return len(self.labels_df)

    def get_weight(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        cls = 0 if self.labels_df.iloc[idx, self.class_iloc] < self.class_threshold else 1
        weight = 1.0 / self.class_freq[cls]
        return weight

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cls = 0 if self.labels_df.iloc[idx, self.class_iloc] < self.class_threshold else 1
        prefix = 'pos' if self.class_mapping[cls] > 0 else 'neg'
        if not self.use_prefix: prefix = ''

        img_name = os.path.join(self.root_dir, prefix, self.labels_df.iloc[idx, 0] + self.file_type)
        img = cv2.imread(img_name)
        assert img is not None, f'Image {img_name} has to exist'

        sample = {'orig_img': img, 'image': img, 'label': self.class_mapping[cls], 'max_fs': self.labels_df.iloc[idx, self.class_iloc],
                  'name': os.path.basename(img_name), 'eye': self.labels_df.iloc[idx, 0][:-1]}
        if self.augs:
            sample['image'] = self.augs(image=img)['image']
        return sample


class SegmentsDataset(RetinaDataset):
    def __init__(self, csv_file, root_dir, file_type='.jpg', class_iloc=1, augmentations=None, use_prefix=False,
                 thresh=1, segments=4):
        super().__init__(csv_file, root_dir, file_type, class_iloc, augmentations, use_prefix, thresh)
        self.seg_df = pd.DataFrame(columns=self.labels_df.columns)
        for row in self.labels_df.itertuples(index=False):
            for i in range(segments):
                self.seg_df = self.seg_df.append({
                    self.labels_df.columns[0]: row[0] + str(i+1),
                    self.labels_df.columns[1]: row[1]
                }, ignore_index=True)
        print(f'Original dataframe length: {len(self.labels_df)}, expanded length: {len(self.seg_df)}')
        self.labels_df = self.seg_df


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
        #A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.3),
        # border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
        A.OneOf(
            [A.GaussNoise(p=0.5), A.ISONoise(p=0.5), A.IAAAdditiveGaussianNoise(p=0.25), A.MultiplicativeNoise(p=0.25)],
            p=0.3),
        A.OneOf([A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5), A.GridDistortion(p=0.5)], p=0.3),
        A.OneOf([A.HueSaturationValue(p=0.5), A.ToGray(p=0.5), A.RGBShift(p=0.5)], p=0.3),
        A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2), A.RandomGamma()], p=0.3),
        A.Normalize(always_apply=True, p=1.0),
        ToTensorV2(always_apply=True, p=1.0)
    ], p=1.0)
    return pipe
