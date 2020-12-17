from typing import Callable

import os
from os.path import join
import numpy as np
import albumentations as A
import cv2
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
        :param augmentations: albumentation data augmentation
        :param use_prefix: data folder contains subfolders for classes (pos / neg)
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

        sample = {'orig_img': img, 'image': img, 'label': self.class_mapping[cls],
                  'max_fs': self.labels_df.iloc[idx, self.class_iloc],
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
                    self.labels_df.columns[0]: row[0] + str(i + 1),
                    self.labels_df.columns[1]: row[1]
                }, ignore_index=True)
        print(f'Original dataframe length: {len(self.labels_df)}, expanded length: {len(self.seg_df)}')
        self.labels_df = self.seg_df


class RetinaBagDataset(RetinaDataset):
    def __init__(self, csv_file, root_dir, file_type='.jpg', class_iloc=1, augmentations=None, use_prefix=False,
                 thresh=1, max_bag_size=100, segment_size=399, exclude_black_th=0.5):
        super().__init__(csv_file, root_dir, file_type, class_iloc, augmentations, use_prefix, thresh)
        self.max_bag_size = max_bag_size
        self.segment_size = segment_size
        self.exclude_black_th = exclude_black_th
        self.bags = self._create_bags()

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        assert not torch.is_tensor(idx)
        bag = self.bags[idx]

        sample = {'frames': [], 'label': bag['label'], 'name': bag['name']}
        eye_img = cv2.imread(os.path.join(self.root_dir, f'{bag["name"]}{self.file_type}'))
        # Apply augmentations BEFORE segmentation

        for y in range(0, bag['h'], self.segment_size):
            for x in range(0, bag['w'], self.segment_size):
                segment = eye_img[y:y + self.segment_size, x:x + self.segment_size]
                if segment.shape[0] * segment.shape[1] != self.segment_size ** 2:
                    continue
                count_non_black_px = cv2.countNonZero(cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY))
                if count_non_black_px / self.segment_size ** 2 < self.exclude_black_th:
                    continue
                segment = self.augs(image=segment)['image'] if self.augs else segment
                sample['frames'].append(segment)
                #sample['pos'].append((y, x, self.segment_size))
        max_count_segments = (bag['h'] // self.segment_size) * (bag['w'] // self.segment_size)
        #print(f'Segmentation excluded {max_count_segments - len(sample["frames"])} segments')
        sample['frames'] = torch.stack(sample['frames']) if self.augs else np.stack(sample['frames'])
        return sample

    def _create_bags(self):
        bags = []
        for row in self.labels_df.itertuples(index=False):
            bag_label = 0 if row[self.class_iloc] < self.class_threshold else 1
            bag_label = self.class_mapping[bag_label]
            bag_name = row[0]
            bag_img = cv2.imread(join(self.root_dir, bag_name + self.file_type))
            bag_height, bag_width = bag_img.shape[:2]
            bags.append({'name': bag_name, 'label': bag_label, 'w': bag_width, 'h': bag_height})

        return bags


########################## Dataset Helper Methods #########################
def get_validation_pipeline(image_size, crop_size, mode='default'):
    return A.Compose([
        A.NoOp() if mode == 'mil' else A.Resize(image_size, image_size, always_apply=True, p=1.0),
        A.CenterCrop(crop_size, crop_size, always_apply=True, p=1.0),
        A.Normalize(always_apply=True, p=1.0),
        ToTensorV2(always_apply=True, p=1.0)
    ], p=1.0)


def get_training_pipeline(image_size, crop_size, mode='default', strength=1.0):
    pipe = A.Compose([
        A.NoOp() if mode == 'mil' else A.Resize(image_size, image_size, always_apply=True, p=1.0),
        A.RandomCrop(crop_size, crop_size, always_apply=True, p=1.0),
        A.HorizontalFlip(p=0.5*strength),
        # A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.3*strength),
        # border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
        A.OneOf([A.GaussNoise(p=0.5), A.ISONoise(p=0.5), A.IAAAdditiveGaussianNoise(p=0.25),
                 A.MultiplicativeNoise(p=0.25)], p=0.3*strength),
        A.OneOf([A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5), A.GridDistortion(p=0.5)],
                p=0.3*strength),
        A.OneOf([A.HueSaturationValue(p=0.5), A.ToGray(p=0.5), A.RGBShift(p=0.5)], p=0.3*strength),
        A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2), A.RandomGamma()],
                p=0.3*strength),
        A.Normalize(always_apply=True, p=1.0),
        ToTensorV2(always_apply=True, p=1.0)
    ], p=1.0)
    return pipe


def get_dataset(dataset: Callable, base_name: str, hp: dict, aug_pipeline_train: A.Compose, aug_pipeline_val: A.Compose,
                num_workers: int):
    set_names = ('train', 'val')
    train_dataset = dataset(join(base_name, 'labels_train.csv'), join(base_name, set_names[0]),
                            augmentations=aug_pipeline_train, file_type='.jpg', use_prefix=False,
                            class_iloc=1,
                            thresh=hp['class_threshold'], segment_size=hp['crop_size'], exclude_black_th=hp['black_threshold'])
    val_dataset = dataset(join(base_name, 'labels_val.csv'), join(base_name, set_names[1]),
                          augmentations=aug_pipeline_val, file_type='.jpg', use_prefix=False, class_iloc=1,
                          thresh=hp['class_threshold'], segment_size=hp['crop_size'], exclude_black_th=hp['black_threshold'])
    sample_weights = [train_dataset.get_weight(i) for i in range(len(train_dataset))]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=False,
                                               sampler=sampler, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hp['batch_size'], shuffle=False,
                                             num_workers=num_workers)
    print(f'Dataset ({dataset.__name__}) info:\n'
          f' Train size: {len(train_dataset)},\n'
          f' Validation size: {len(val_dataset)}')
    return train_loader, val_loader


if __name__ == '__main__':
    pass
