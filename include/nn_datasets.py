import os
from collections import Counter
from os.path import join
import random
import cv2
import nn_utils
import numpy as np
import pandas as pd
import torch
import utils as utl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from nn_utils import get_video_desc
from torch.utils.data import Dataset


class RetinaDataset(Dataset):
    def __init__(self, csv_file, root_dir, file_type='.png', balance_ratio=1.0, transform=None, augmentations=None,
                 use_prefix=False, boost_frames=1.0, occur_balance=False):
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
        self.grade_count = Counter(
            [get_video_desc(name)['eye_id'] for name in self.labels_df['image'].tolist()]) if occur_balance else None
        self.root_dir = root_dir
        self.file_type = file_type
        self.transform = transform
        self.augs = augmentations
        self.ratio = balance_ratio
        self.use_prefix = use_prefix
        self.boost = boost_frames
        self.occur_balance = occur_balance
        assert transform is None or augmentations is None
        assert (boost_frames > 1.0 and len(self.labels_df.columns) > 2) or boost_frames == 1.0

    def __len__(self):
        return len(self.labels_df)

    def get_weight(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        severity = self.labels_df.iloc[idx, 1]
        weight = self.ratio if severity == 0 else 1.0
        if self.occur_balance: weight /= self.grade_count[get_video_desc(self.labels_df.iloc[idx, 0])['eye_id']]
        if self.boost > 1.0 and severity == 1 and self.labels_df.iloc[idx, 2] == 1:
            weight *= (1. + self.labels_df.iloc[idx, 3])
        return weight

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        severity = self.labels_df.iloc[idx, 1]

        if self.use_prefix:
            prefix = 'pos' if severity == 1 else 'neg'
        else:
            prefix = ''
        img_name = os.path.join(self.root_dir, prefix, self.labels_df.iloc[idx, 0] + self.file_type)
        img = cv2.imread(img_name)
        assert img is not None, f'Image {img_name} has to exist'
        # image = image[:,:,[2, 1, 0]]

        sample = {'image': img, 'label': severity,
                  'eye_id': get_video_desc(self.labels_df.iloc[idx, 0], only_eye=True)['eye_id'],
                  'name': self.labels_df.iloc[idx, 0]}
        if self.transform:
            sample['image'] = img[:, :, [2, 1, 0]]
            sample['image'] = self.transform(sample['image'])
        if self.augs:
            sample['image'] = self.augs(image=img)['image']
        return sample


class MultiChannelRetinaDataset(RetinaDataset):
    def __init__(self, csv_file, root_dir, file_type='.png', balance_ratio=1.0, augmentations=None, use_prefix=False,
                 processed_suffix='_nn'):
        super().__init__(csv_file, root_dir, file_type, balance_ratio, None, augmentations, use_prefix)
        self.suffix = processed_suffix

    def __getitem__(self, idx):
        assert not torch.is_tensor(idx)
        severity = self.labels_df.iloc[idx, 1]
        if self.use_prefix:
            prefix = 'pos' if severity == 1 else 'neg'
        else:
            prefix = ''
        img_name = os.path.join(self.root_dir, prefix, self.labels_df.iloc[idx, 0] + self.file_type)
        processed_name = os.path.join(self.root_dir + self.suffix, prefix, self.labels_df.iloc[idx, 0] + self.file_type)
        img = cv2.imread(img_name)
        processed_img = cv2.imread(processed_name)
        sample = {'image': None, 'label': severity,
                  'eye_id': get_video_desc(self.labels_df.iloc[idx, 0], only_eye=True)['eye_id']}
        if self.augs:
            img = self.augs(image=img)['image']
            processed_img = self.augs(image=processed_img)['image']
            sample['image'] = torch.cat([img, processed_img])
        return sample


class FiveCropRetinaDataset(Dataset):
    def __init__(self, csv_file, root_dir, size, file_type='.png', balance_ratio=1.0, augmentations=None,
                 use_prefix=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.file_type = file_type
        self.augs = augmentations
        self.ratio = balance_ratio
        self.use_prefix = use_prefix
        self.num_crops = 5
        self.size = size

    def __len__(self):
        return len(self.labels_df) * self.num_crops

    def get_weight(self, idx):
        assert not torch.is_tensor(idx)
        severity = self.labels_df.iloc[idx // self.num_crops, 1]
        return self.ratio if severity == 0 else 1.0

    def __getitem__(self, idx):
        assert not torch.is_tensor(idx)

        image_idx = idx // self.num_crops
        crop_idx = idx % self.num_crops

        severity = self.labels_df.iloc[image_idx, 1]

        if self.use_prefix:
            prefix = 'pos' if severity == 1 else 'neg'
        else:
            prefix = ''

        img_name = os.path.join(self.root_dir, prefix, self.labels_df.iloc[image_idx, 0] + self.file_type)
        img = cv2.imread(img_name)
        # image = image[:,:,[2, 1, 0]]

        sample = {'image': img, 'label': severity, 'image_idx': image_idx}
        if self.augs:
            img = cv2.resize(img, (self.size[0], self.size[0]))
            img = utl.do_five_crop(img, self.size[1], self.size[1], crop_idx)
            sample['image'] = self.augs(image=img)['image']
        return sample


class SnippetDataset(Dataset):
    def __init__(self, csv_file, root_dir, num_frames=20, file_type='.png', balance_ratio=1.0, augmentations=None):
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.file_type = file_type
        self.augs = augmentations
        self.ratio = balance_ratio
        self.num_frames = num_frames

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        assert not torch.is_tensor(idx)

        image_idx = idx

        severity = self.labels_df.iloc[image_idx, 1]
        prefix = 'pos' if severity == 1 else 'neg'

        snip_name = self.labels_df.iloc[image_idx, 0]
        video_desc = get_video_desc(snip_name)

        video_all_frames = [f for f in os.listdir(os.path.join(self.root_dir, prefix)) if
                            video_desc['eye_id'] == get_video_desc(f)['eye_id']]
        # if len(frame_index) - 1 < video_index:
        #    print('Problem with video ', video_name, video_index)

        # frame_names = sorted([f for f in files if video_desc['snippet_id'] == get_video_desc(f)['snippet_id']], key=lambda n: get_video_desc(n)['frame_id'])
        selection = np.random.randint(0, len(video_all_frames), self.num_frames)  # Generate random indicies
        selected_frames = [video_all_frames[idx] for idx in selection]

        sample = {'frames': [], 'label': severity, 'name': video_desc['eye_id'][:5]}
        for name in selected_frames:
            img = cv2.imread(os.path.join(self.root_dir, prefix, name))
            img = self.augs(image=img)['image'] if self.augs else img
            sample['frames'].append(img)

        sample['frames'] = torch.stack(sample['frames']) if self.augs else np.stack(sample['frames'])
        return sample

    def get_weight(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        severity = self.labels_df.iloc[idx, 1]
        return self.ratio if severity == 0 else 1.0


class SnippetDataset2(Dataset):
    def __init__(self, csv_file, root_dir, num_frames=20, file_type='.png', balance_ratio=1.0, augmentations=None, bagging_strategy='snippet'):
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.file_type = file_type
        self.augs = augmentations
        self.ratio = balance_ratio
        self.num_frames = num_frames
        self.bag_strategy = bagging_strategy

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        assert not torch.is_tensor(idx)

        image_idx = idx

        severity = self.labels_df.iloc[image_idx, 1]
        prefix = 'pos' if severity == 1 else 'neg'

        snip_name = self.labels_df.iloc[image_idx, 0]
        snip_desc = get_video_desc(snip_name)

        video_all_frames = [f for f in os.listdir(os.path.join(self.root_dir, prefix)) if
                            snip_desc['eye_id'] == get_video_desc(f)['eye_id']]

        if self.bag_strategy == 'snippet':
            snippet_frames = sorted(
                [f for f in video_all_frames if snip_desc['snippet_id'] == get_video_desc(f)['snippet_id']],
                key=lambda n: get_video_desc(n)['frame_id'])
            selected_frames = snippet_frames
        elif self.bag_strategy == 'random':
            selection = np.random.randint(0, len(video_all_frames), self.num_frames)  # Generate random indicies
            selected_frames = [video_all_frames[idx] for idx in selection]
        else:
            snippet_list = list(set([get_video_desc(f)['snippet_id'] for f in video_all_frames]))
            selected_snippets = np.random.choice(snippet_list, self.num_frames)
            selected_frames = []
            for snippet in selected_snippets:
                selected_frame_idx = np.random.randint(0, 20)
                selected_frames.append(
                    [f for f in video_all_frames if int(snippet) == get_video_desc(f)['snippet_id']][selected_frame_idx])

        sample = {'frames': {}, 'label': severity, 'name': snip_desc['eye_id'][:5]}
        for i, name in enumerate(selected_frames):
            img = cv2.imread(os.path.join(self.root_dir, prefix, name))
            sample_name = 'image' + (str(i) if i != 0 else '')
            sample['frames'][sample_name] = img

        self.augs.add_targets({key: 'image' for key in list(sample['frames'].keys())[1:]})
        sample['frames'] = list(self.augs(**sample['frames']).values())
        sample['frames'] = torch.stack(sample['frames']) if self.augs else np.stack(sample['frames'])
        return sample

    def get_weight(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        severity = self.labels_df.iloc[idx, 1]
        return self.ratio if severity == 0 else 1.0


class PaxosBags(Dataset):
    def __init__(self, csv_file, root_dir, augmentations = None, balance_ratio=1.0, max_bag_size=100):
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.augs = augmentations
        self.ratio = balance_ratio
        self.max_bag_size = max_bag_size
        self.occurrences = {}
        for row in self.labels_df.itertuples():
            eye_id = nn_utils.get_video_desc(row.image)['eye_id']
            entry = self.occurrences.get(eye_id)
            self.occurrences[eye_id] = self.occurrences[eye_id] + 1 if entry else 1
        self.bags = self._create_bags()

    def __len__(self):
        num_bags = 0
        for eye in self.occurrences.values():
            num_bags += (1 if eye <= self.max_bag_size else int(np.ceil(eye / self.max_bag_size)))
        return num_bags

    def __getitem__(self, idx):
        assert not torch.is_tensor(idx)
        bag = self.bags[idx]
        prefix = 'pos' if bag['label'] == 1 else 'neg'

        sample = {'frames': [], 'label': bag['label'], 'name': bag['name'][:5]}
        for name in bag['frames']:
            img = cv2.imread(os.path.join(self.root_dir, prefix, name))
            img = self.augs(image=img)['image'] if self.augs else img
            sample['frames'].append(img)

        sample['frames'] = torch.stack(sample['frames']) if self.augs else np.stack(sample['frames'])
        return sample

    def get_weight(self, idx):
        assert not torch.is_tensor(idx)
        bag = self.bags[idx]
        return self.ratio if bag['label'] == 0 else 1.0

    def _create_bags(self):
        bags = []
        for eye, value in self.occurrences.items():
            bag_label = self.labels_df[self.labels_df.image.str.contains(eye)].iloc[0].level
            prefix = 'pos' if bag_label == 1 else 'neg'
            eye_frames = [f for f in os.listdir(join(self.root_dir, prefix)) if get_video_desc(f)['eye_id'] == eye]
            #random.shuffle(eye_frames)
            if value <= self.max_bag_size:
                bags.append({'frames': eye_frames, 'label': bag_label, 'name': f'{eye}_{0}', 'shortname': eye})
            else:
                for i, start_idx in enumerate(range(0, len(eye_frames), self.max_bag_size)):
                    bags.append({'frames': eye_frames[start_idx:start_idx+self.max_bag_size], 'label': bag_label, 'name': f'{eye}_{i}'})
        print('Created bags: ', len(bags), self.occurrences)
        return bags


########################## Dataset Helper Methods #########################
def get_validation_pipeline(image_size, crop_size):
    return A.Compose([
        A.Resize(image_size, image_size, always_apply=True, p=1.0),
        A.CenterCrop(crop_size, crop_size, always_apply=True, p=1.0),
        A.Normalize(always_apply=True, p=1.0),
        ToTensorV2(always_apply=True, p=1.0)
    ], p=1.0)


def get_training_pipeline(image_size, crop_size, strength=0):
    if strength == 0:
        return A.Compose([
            A.Resize(image_size, image_size, always_apply=True, p=1.0),
            A.RandomCrop(crop_size, crop_size, always_apply=True, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.CoarseDropout(min_holes=1, max_holes=3, max_width=75, max_height=75, min_width=25, min_height=25, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT,
                               value=0,
                               p=0.5),
            A.OneOf([A.HueSaturationValue(p=0.5), A.ToGray(p=0.5), A.RGBShift(p=0.5)], p=0.3),
            A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2), A.RandomGamma()], p=0.5),
            A.Normalize(always_apply=True, p=1.0),
            ToTensorV2(always_apply=True, p=1.0)
        ], p=1.0)
    else:
        return None
