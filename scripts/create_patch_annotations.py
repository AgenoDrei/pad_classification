from typing import Tuple
import pandas as pd
import numpy as np
import click
import json
import pickle
import cv2
import os
import matplotlib.cm as cm
from shapely.geometry import Polygon


dataset = {}
width, height = (9, 8)


def create_patches(patch_size, rows, columns):
    patches = []
    for y in range(rows):
        for x in range(columns):
            lt = (x * patch_size, y * patch_size)
            rt = ((x + 1) * patch_size, y * patch_size)
            lb = (x * patch_size, (y+1) * patch_size)
            rb = ((x+1) * patch_size, (y+1) * patch_size)
            p = Polygon([lt, rt, rb, lb])
            patch = {'shape': p, 'x': x, 'y': y, "visible_classes": []}
            patches.append(patch)
    return patches


def process_annotation(file_name, region, annotation):
    entry = dataset.get(file_name)
    structure = None
    if not entry:
        dataset[file_name] = create_patches(399, height, width)
        entry = dataset[file_name]
    if region["name"] == 'circle':
        print('Circle annotation detected')
        lt = (region["cx"] - region["r"]//2.5, region["cy"] - region["r"]//2.5)
        lb = (region["cx"] - region["r"]//2.5, region["cy"] + region["r"]//2.5)
        rt = (region["cx"] + region["r"]//2.5, region["cy"] - region["r"]//2.5)
        rb = (region["cx"] - region["r"]//2.5, region["cy"] + region["r"]//2.5)
        structure = Polygon([lt, rt, rb, lb])
    elif region["name"] == 'polygon':
        print('Polygon annotation detected')
        points = [(x, y) for x,y in zip(region["all_points_x"], region["all_points_y"])]
        structure = Polygon(points)

    for e in entry:
        cell = e["shape"]
        if cell.intersects(structure):
            e["visible_classes"].extend([int(k) for k in annotation["Anatomical structure"].keys() if k])


def write_annotations(patch_list, path):
    anno_grid = []
    for p in patch_list:
        if len(p['visible_classes']) > 0:
            anno_grid.append(cm.viridis(p['visible_classes'][0] / 10)[:3])
        else:
            anno_grid.append((0, 0, 0))
    anno_grid = np.array(anno_grid, ndmin=3)
    anno_grid = anno_grid.reshape((height, width, 3))
    anno_img = cv2.resize(anno_grid, (width * 399, height * 399))
    anno_img = cv2.normalize(anno_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite(path, anno_img)


@click.command()
@click.option('--input_path', '-i', help='Input path')
@click.option('--output_path', '-o', help='Output path for patch annotation')
@click.option('--annotations', '-a', help='Via annotation file')
def run(input_path, output_path, annotations):
    df = pd.read_csv(annotations)

    for i, row in df.iterrows():
        print(f'Processing annotation {row["region_id"]} for {row["filename"]}')
        region = json.loads(row["region_shape_attributes"])
        annotation = json.loads(row["region_attributes"])
        process_annotation(row["filename"], region, annotation)

    # print(dataset)
    # write_annotations(dataset[cur_file], output_path)

    # Exporting annotations
    output_file = open(os.path.join(output_path, 'output.pkl'), 'wb')
    pickle.dump(dataset, output_file)
    output_file_json = open(os.path.join(output_path, 'output.json'), 'w')
    json.dump(dataset, output_file_json, default=lambda o: '<not serializable>')


if __name__ == '__main__':
    run()
