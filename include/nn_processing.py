import os
from shutil import copy

import numpy as np
import cv2
import torch
import utils as utl
from albumentations import DualTransform
from joblib import Parallel, delayed
from skimage import exposure, morphology, img_as_ubyte, img_as_float


class CenterCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        image = image[top: top + new_h, left: left + new_w]
        return image


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        h, w, _ = image.shape
        #image = trans.resize(image, (h, w))

        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h, left: left + new_w]
        return image


class RandomNormalCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        h, w, _ = image.shape
        #image = trans.resize(image, (h, w))

        new_h, new_w = self.output_size
        mean_h, mean_w = (h - new_h) // 2, (w - new_w) // 2
        std_h, std_w = mean_h * 0.25, mean_w * 0.25

        top_rand = np.random.normal(mean_h, std_h)
        top = top_rand if top_rand < h - new_h else h - new_h - 1
        top = top if top > 0 else 0
        left_rand = np.random.normal(mean_w, std_w)
        left = left_rand if left_rand < w - new_w else w - new_w - 1
        left = left if left > 0 else 0
        #top = np.random.randint(0, h - new_h)
        #left = np.random.randint(0, w - new_w)
        image = image[int(top): int(top) + new_h, int(left): int(left) + new_w]
        return image


class Flip(object):
    def __init__(self, probability):
        assert isinstance(probability, float)
        self.prob = probability

    def __call__(self, image):
        if np.random.rand() < self.prob:
            image = cv2.flip(image, 1)
        return image


class Blur(object):
    def __init__(self, probability):
        assert isinstance(probability, float)
        self.prob = probability

    def __call__(self, image):
        if np.random.rand() < self.prob:
            image = cv2.GaussianBlur(image, (5, 5) ,0)
        return image


class EnhanceContrast(object):
    def __init__(self, probability):
        assert isinstance(probability, float)
        self.prob = probability

    def __call__(self, image):
        if np.random.rand() < self.prob:
            image = image[:, :, [2, 1, 0]]
            image = utl.enhance_contrast_image(image, clip_limit=np.random.randint(2, 5))
            image = image[:, :, [2, 1, 0]]
        return image


class ToTensor(object):
    def __call__(self, image):
        # swap color axis because, DOES NOT NORMALIZE RIGHT NOW!
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)


class RandomFiveCrop(DualTransform):
    def __init__(self, height, width, always_apply=True, p=1):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def apply(self, img, **params):
        state = np.random.randint(0, 5)
        return utl.do_five_crop(img, self.height, self.width, state=state)

    def get_transform_init_args_names(self):
        return 'height', 'width'

    def get_params_dependent_on_targets(self, params):
        pass


class ThresholdGlare(DualTransform):
    def __init__(self, always_apply=True, p=1, thresh=0.85):
        super().__init__(always_apply, p)
        self.thresh = thresh

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def apply(self, img, **params):
        img2 = utl.enhance_contrast_image(img)
        img2 = img2[:, :, 1]

        mask = img2 > self.thresh
        mask = cv2.dilate(img_as_ubyte(mask), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))

        return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    def get_transform_init_args_names(self):
        return 'thresh'

    def get_params_dependent_on_targets(self, params):
        pass


def copy_corresponding_files(filter_str: str, file_list: list, base_path: str, prefix: str, set_str: str = '', copy_mode: bool = True):
    """
    Move all files in file list, that have the filter string in them
    :param filter_str: Name that has to be in filename to be moved
    :param file_list: list of all files that could be moved
    :param base_path: absolute path of all files
    :param prefix: pos or neg
    :param set_str: name of set (train/val) [Optional]
    :param copy_mode: Copy (True) or move (False)
    :return:
    """
    method = copy if copy_mode else os.rename
    Parallel(n_jobs=-1, verbose=0)(delayed(method)(f, os.path.join(base_path, set_str, prefix, os.path.basename(f))) for f in file_list if filter_str in f)
