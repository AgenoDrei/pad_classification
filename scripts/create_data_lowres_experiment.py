import os
import cv2
import click
import shutil
import os.path as osp


INTER_MODE = cv2.INTER_CUBIC


@click.command()
@click.option('--input_path', '-i', help='Input path', required=True)
@click.option('--output_path', '-o', help='Output path', required=True)
@click.option('--resolution', '-r', help='Downsize resolution for images', required=False, default=399)
@click.option('--upsample/--downsample', default=False)
@click.option('--pyramids/--nopyramids', help='Use gaussian pyramids for downsampling to avoid artifacts', default=False)
def run(input_path, output_path, resolution=399, upsample=False, pyramids=False):
    os.mkdir(output_path)
    for (root, dirs, files) in os.walk(input_path):
        print('Working on folder: ', root)
        rel_path = osp.relpath(root, input_path)
        for d in dirs:
            os.mkdir(osp.join(output_path, rel_path, d))
        for f in files:
            if osp.splitext(f)[1] == '.csv':
                shutil.copy2(osp.join(root, f), osp.join(output_path, rel_path, f))
                continue
            if osp.splitext(f)[1] != '.png' and osp.splitext(f)[1] != '.jpg':
                continue
            img = cv2.imread(osp.join(root, f))
            height, width = img.shape[:2]
            
            if pyramids:
                orig_shape = img.shape[:2]
                while img.shape[0] > resolution:
                    img = cv2.pyrDown(img)
                new_img = img
                if upsample:
                    while new_img.shape[0] * 2 <= orig_shape[0]:
                        new_img = cv2.pyrUp(new_img)
            else:
                new_img = cv2.resize(img, (resolution, resolution), interpolation=INTER_MODE)
                if upsample: new_img = cv2.resize(new_img, (width, height), interpolation=INTER_MODE)
            
            cv2.imwrite(osp.join(output_path, rel_path, f), new_img)


if __name__ == '__main__':
    run()
