import os
import cv2
import click
import pickle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu
from tqdm import tqdm


id2desc = {0: "sup. temporal artery", 1: "sup. temporal vein", 2: "sup. nasal artery", 3: "sup. nasal vein",
           4: "inf. temporal artery", 5: "inf. temporal vein", 6: "inf. nasal artery", 7: "inf. nasal vein",
           8: "macula", 9: "optic disc"}


def fix_zero_att_weights(weights, annotations, cur_img_id, data_path, segment_size=399, black_th=0.5):
    img = cv2.imread(os.path.join(data_path, cur_img_id))
    seg_count = 0

    for y in range(0, img.shape[0], segment_size):
        for x in range(0, img.shape[1], segment_size):
            segment = img[y:y + segment_size, x:x + segment_size]
            if segment.shape[0] * segment.shape[1] != segment_size ** 2:
                continue
            count_non_black_px = cv2.countNonZero(cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY))
            if count_non_black_px / segment_size ** 2 < black_th:
                #weights = np.insert(weights, seg_count, 0.0)
                #annotations = np.delete(annotations, seg_count)
                del annotations[seg_count]
                seg_count -= 1
            seg_count += 1
    return annotations


def enrich_annotations(annotations, df, data):
    for i, row in tqdm(df.iterrows(), total=len(df), desc='Enrich annotations'):
        filename = row['eye_id'] + '.jpg'
        sample_patches = annotations[filename]
        weights = eval(row['attention'])
        annotations[filename] = fix_zero_att_weights(weights, sample_patches, filename, data)
        # print(len(sample_patches), len(weights))

        for j in range(len(weights)):
            sample_patches[j]['weight'] = weights[j]
    return annotations


def create_groups(anno, group_idx=0):
    test_group = []
    control_group = []

    for sample in anno.values():
        for cell in sample:
            if group_idx in cell['visible_classes']:
                test_group.append(cell['weight'])
            else:
                control_group.append(cell['weight'])

    return test_group, control_group


@click.command()
@click.option('--attention_weights', '-w', help='Path to MIL results containing weights for images')
@click.option('--annotations', '-a', help='Path to VIA annotation file')
@click.option('--data', '-d', help='Path to the image dataset folder')
def run(attention_weights, annotations, data):
    df = pd.read_csv(attention_weights)

    file = open(annotations, 'rb')
    anno = pickle.load(file)
    file.close()

    anno = enrich_annotations(anno, df, data)
    #print(anno["HA1044L.jpg"])

    for i in range(0, 10):
        test_group, control_group = create_groups(anno, group_idx=i)
        stat, p = mannwhitneyu(np.array(test_group), np.array(control_group), alternative='greater')
        print(f'P-value for U-test for class {id2desc[i]}: {p} [{len(test_group)}/{len(control_group)}]')


if __name__ == '__main__':
    run()