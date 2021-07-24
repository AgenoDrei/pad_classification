import os.path
import click
import cv2
import numpy as np
from matplotlib import cm, colors


@click.command()
@click.option('--original-image', '-i', help='Path to original image', required=True)
@click.option('--resolutions', '-r', help='Path to images compared to the original', multiple=True, required=True)
@click.option('--output', '-o', help='Output path', default='/tmp/')
def run(original_image, resolutions, output):
    orig_image = cv2.imread(original_image, cv2.IMREAD_GRAYSCALE)
    orig_shape = orig_image.shape[:2]
    orig_image = np.float32(orig_image)
    print(f'Original shape of the image {original_image} is {orig_shape}, type: {orig_image.dtype}')

    for path in resolutions:
        cmp_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        cmp_img = np.float32(cmp_img)
        print(f'Generate diff images and map for {path} with the resolution {cmp_img.shape}')

        # cmp_img = cv2.resize(cmp_img, (orig_shape[1], orig_shape[0]))
        diff_img = (orig_image - cmp_img) # cv2.absdiff(orig_image, cmp_img) # cv2.absdiff(orig_image, cmp_img) # orig_image - cmp_img
        print(f'Min value: {np.amin(diff_img)}, max value: {np.amax(diff_img)}, type: {diff_img.dtype}')
        # norm_coef = 255
        # diff_img /= norm_coef
        # diff_img = (diff_img + 255) // 2
        # diff_img = np.uint8(diff_img)

        #colormap = cm.RdBu((diff_img) / (diff_img.max() - diff_img.min()) + 0.5)
        mapper = cm.get_cmap('PuOr')
        norm = colors.Normalize(vmin=diff_img.min()+50, vmax=np.abs(diff_img.min())-50, clip=True)
        colormap = mapper(norm(diff_img))
        colormap = np.uint8(colormap * 256)

        print(f'Min value: {np.amin(diff_img)}, max value: {np.amax(diff_img)}, type: {diff_img.dtype}')
        print(f'Min value: {np.amin(colormap)}, max value: {np.amax(colormap)}, type: {colormap.dtype}')
        #colormap = cv2.applyColorMap(diff_img, cv2.COLORMAP_HOT)
        cv2.imwrite(output + os.path.basename(path), colormap)

if __name__ == '__main__':
    run()
