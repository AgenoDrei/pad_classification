import os.path
import click
import cv2
import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

@click.command()
@click.option('--original-image', '-i', help='Path to original image', required=True)
@click.option('--resolutions', '-r', help='Path to images compared to the original', multiple=True, required=True)
@click.option('--output', '-o', help='Output path', default='/tmp/')
def run(original_image, resolutions, output):
    print("Images should be prepared according to these steps: ")
    print(" 1) All images should have the same size")
    print(" 2) If possible, align the images using the following tool: align_image_stack -a aligned -v *.jpg --corr=0.8 -c 32 --gpu")
    print(" 3) Choose between a signed diverging OR sequential mode.")
    input("Enter y) to continue\n")

    orig_image = cv2.imread(original_image, cv2.IMREAD_GRAYSCALE)
    orig_shape = orig_image.shape[:2]
    orig_image = np.float32(orig_image)
    print(f'Original shape of the image {original_image} is {orig_shape}, type: {orig_image.dtype}')
    diff_maps = []
    cmap = "gist_yarg" # "bwr"
    mode = "sequential"   # "diverging" # "sequential"

    for path in resolutions:
        cmp_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        cmp_img = np.float32(cmp_img)
        print(f'Generate diff images and map for {path} with the resolution {cmp_img.shape}')

        # cmp_img = cv2.resize(cmp_img, (orig_shape[1], orig_shape[0]))
        #diff_img = (orig_image - cmp_img)
        if mode == "sequential":
            diff_img = cv2.absdiff(orig_image, cmp_img) 
        elif mode == "diverging":
            diff_img = orig_image - cmp_img
        print(f'Min value: {np.amin(diff_img)}, max value: {np.amax(diff_img)}, type: {diff_img.dtype}')
        norm_coef = 98
        diff_img = np.clip(diff_img, -98, 98)
        diff_img /= norm_coef

        #colormap = cm.RdBu((diff_img) / (diff_img.max() - diff_img.min()) + 0.5)
        #mapper = cm.get_cmap(cmap)
            
        #if mode == "sequential":
        #    norm = colors.Normalize(vmin=0, vmax=136, clip=True)
        #elif mode == "diverging":
        #    norm = colors.Normalize(vmin=diff_img.min()+0, vmax=np.abs(diff_img.min())-0, clip=False)
        #colormap = norm(diff_img)
        #colormap = mapper(colormap)
        #colormap = np.uint8(colormap * 256)
    
        print(f'Min value: {np.amin(diff_img)}, max value: {np.amax(diff_img)}, type: {diff_img.dtype}')
        #print(f'Min value: {np.amin(colormap)}, max value: {np.amax(colormap)}, type: {colormap.dtype}')
        #colormap = cv2.applyColorMap(diff_img, cv2.COLORMAP_HOT)
        #cv2.imwrite(output + os.path.basename(path), colormap)
        diff_maps.append(diff_img[:])
    
    vmin = 0 if mode == "sequential" else -1
    fig = plt.figure(figsize=(15, 4))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    im = ax.imshow(diff_maps[0], cmap=cmap, vmin=vmin, vmax=1)
    im2 = ax2.imshow(diff_maps[1], cmap=cmap, vmin=vmin, vmax=1)
    
    fig.subplots_adjust(right=0.9)
    #cbar_ax = fig.add_axes([0.925, 0.075, 0.05, 0.7])
    cbar = fig.colorbar(im2, ax=[ax, ax2])
    #cbar = ax.colorbar(fig)
    #cbar.ax.set_ylabel("legend", rotation=-90, va="bottom")
    ax.axis('off')
    ax2.axis('off')
    plt.show()
    #cv2.waitKey(0)
    plt.savefig('colormap.png')


if __name__ == '__main__':
    run()
