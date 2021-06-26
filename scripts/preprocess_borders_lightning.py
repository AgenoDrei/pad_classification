import joblib as j
import argparse
import cv2
import os
import numpy as np
from os.path import join


def crop_image_from_gray(img, tol=7):
    """
    Crop out black borders
    https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def preprocess(img_path, input_path, output_path, light=False, sigma=10):
    image = cv2.imread(join(input_path, img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Ben Grahams method to improve lighning
    if light:
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigma), -4, 128)

    # Circle crop
    height, width, depth = image.shape
    largest_side = np.max((height, width))
    image = cv2.resize(image, (largest_side, largest_side))
    height, width, depth = image.shape
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    image = cv2.bitwise_and(image, image, mask=circle_img)
    image = crop_image_from_gray(image)
    
    image = cv2.resize(image, (1024, 1024))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(join(output_path, img_path), image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    a = argparse.ArgumentParser()
    a.add_argument("--input", help="absolute path to input folder")
    a.add_argument("--output", help="absolute path to output folder")
    a.add_argument("--improve_light", help="apply ben graham light improvements", action="store_true")
    a.add_argument("--recursive", "-r", help="apply recursivly to all directories", action="store_true")
    args = a.parse_args()
    print(args)

    assert os.path.exists(args.input)
    if os.path.exists(args.output):
        os.rmdir(args.output)
    os.mkdir(args.output)

    if not args.recursive:
        j.Parallel(n_jobs=-1, verbose=1)(j.delayed(preprocess)(f, args.input, args.output, light=args.improve_light) for f in os.listdir(args.input))
    else:
        files = os.walk(args.input)
        working_paths = []
        for path in files:
            if path[1] == 0 and path[2] != 0:
                working_paths.append(path[0])
        for path in working_paths:
            print('Processing folder: ', path)
            j.Parallel(n_jobs=-1, verbose=1)(j.delayed(preprocess)(f, join(args.input, path), join(args.output, path), light=args.improve_light) for f in os.listdir(path))
