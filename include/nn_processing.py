import os
import cv2
from shutil import copy
from albumentations import ImageOnlyTransform


class GrahamFilter(ImageOnlyTransform):
    def get_transform_init_args_names(self):
        pass

    def get_params_dependent_on_targets(self, params):
        pass

    def apply(self, img, sigma=10):
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigma), -4, 128)
        return img


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
