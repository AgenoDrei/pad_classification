import os
import cv2
import click
import pickle
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, mannwhitneyu, wilcoxon
from tqdm import tqdm


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
                # weights = np.insert(weights, seg_count, 0.0)
                # annotations = np.delete(annotations, seg_count)
                del annotations[seg_count]
                seg_count -= 1
            seg_count += 1
    return annotations


def enrich_annotations(annotations, df, data, combine=False):
    if os.path.exists('/tmp/annos.pkl'):
        file = open('/tmp/annos.pkl', 'rb')
        anno = pickle.load(file)
        file.close()
        return anno

    for i, row in tqdm(df.iterrows(), total=len(df), desc='Enrich annotations'):
        filename = row['eye_id'] + '.jpg'
        sample_patches = annotations[filename]
        weights = eval(row['attention'])
        weights = [w * len(weights) for w in weights]
        annotations[filename] = fix_zero_att_weights(weights, sample_patches, filename, data)
        # print(len(sample_patches), len(weights))

        for j in range(len(weights)):
            sample_patches[j]['weight'] = weights[j]
            if combine:
                sample_patches[j]['visible_classes'] = \
                    [x - 1 if x == 1 or x == 3 or x == 5 or x == 7 else x for x in sample_patches[j]['visible_classes']]

    output_file = open('/tmp/annos.pkl', 'wb')
    pickle.dump(annotations, output_file)
    output_file.close()
    return annotations


def create_groups(anno, group_idx=0, only_empty=False):
    test_group = []
    control_group = []

    for sample in anno.values():
        for cell in sample:
            if group_idx in cell['visible_classes']:
                test_group.append(cell['weight'])
            else:
                if only_empty:
                    if len(cell['visible_classes']) == 0:
                        control_group.append(cell['weight'])
                else:
                    control_group.append(cell['weight'])

    return test_group, control_group


def create_pooled_groups(anno, id2desc):
    id2desc[10] = 'no_class'
    groups = {k: [] for k in id2desc.keys()}

    for sample in anno.values():
        w_agg = {k: [0.0, 0] for k in id2desc.keys()}
        for cell in sample:
            if len(cell['visible_classes']) == 0:
                w_agg[10][0] += cell['weight']
                w_agg[10][1] += 1
            for v in cell['visible_classes']:
                w_agg[v][0] += cell['weight']
                w_agg[v][1] += 1
        w_agg = {k: w_agg[k][0] / (w_agg[k][1] + 10e-100) for k in id2desc.keys()}
        for k in groups.keys():
            groups[k].append(w_agg[k])
    return groups


def plot_groups(test_group, control_group, path='tmp/output.png'):
    fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_axes([0, 0, 1, 1])
    bp = plt.boxplot([test_group, control_group], showfliers=False)
    plt.xticks([1, 2], ['test', 'control'])
    plt.savefig(path)
    plt.close(fig)


def create_overlap_matrix(annotations, id2desc, path='/tmp/', normalize=False):
    trans = {0: 0, 2: 1, 4: 2, 6: 3, 8: 4, 9: 5}
    ol_mat = np.zeros((len(id2desc.values()), len(id2desc.values())))
    for sample in annotations.values():
        for cell in sample:
            classes = cell['visible_classes']
            if len(classes) == 1:
                if len(id2desc.values()) == 6:
                    ol_mat[trans[classes[0]], trans[classes[0]]] += 1
                else:
                    ol_mat[classes[0], classes[0]] += 1
            elif len(classes) > 1:
                for a, b in list(itertools.product(classes, classes)):
                    print()
                    #if a == b:
                    #    continue
                    if len(id2desc.values()) == 6:
                        ol_mat[trans[a], trans[b]] += 1
                        ol_mat[trans[b], trans[a]] += 1
                    else:
                        ol_mat[a, b] += 1
                        ol_mat[b, a] += 1
    # Normalization along rows
    if normalize:
        diag = np.diagonal(ol_mat)
        with np.errstate(divide='ignore', invalid='ignore'):
            ol_mat = np.nan_to_num(np.true_divide(ol_mat, diag[:, None]))
    fig = plt.figure(figsize=(10, 7))
    plt.yticks(list(range(0, len(id2desc.values()))), id2desc.values())
    plt.xticks(list(range(0, len(id2desc.values()))), [t[:4] for t in id2desc.values()])
    plt.imshow(ol_mat)
    plt.colorbar()
    plt.savefig(path + 'overlap_norm.png')
    plt.close(fig)
    return ol_mat


@click.command()
@click.option('--attention_weights', '-w', help='Path to MIL results containing weights for images')
@click.option('--annotations', '-a', help='Path to VIA annotation file')
@click.option('--data', '-d', help='Path to the image dataset folder')
@click.option('--test', help='Type of statistical test used (u-test / friedman-test / wilcoxon-test)', default='u-test')
@click.option('--combine/--separate', help='Combine vessels into one class', default=False)
@click.option('--normalize/--count', help='Normalize co-occurence', default=False)
def run(attention_weights, annotations, data, test='u-test', combine=False, normalize=False):
    df = pd.read_csv(attention_weights)

    file = open(annotations, 'rb')
    anno = pickle.load(file)
    file.close()

    id2desc = {0: "sup. temporal arcade", 2: "sup. nasal arcade", 4: "inf. temporal arcade", 6: "inf. nasal arcade",
               8: "macula", 9: "optic disc"} if combine else {0: "sup. temporal artery", 1: "sup. temporal vein",
                                                              2: "sup. nasal artery", 3: "sup. nasal vein",
                                                              4: "inf. temporal artery", 5: "inf. temporal vein",
                                                              6: "inf. nasal artery", 7: "inf. nasal vein",
                                                              8: "macula", 9: "optic disc"}
    anno = enrich_annotations(anno, df, data, combine=combine)
    overlap = create_overlap_matrix(anno, id2desc, normalize=normalize)

    if test == 'u-test':
        for i in id2desc.keys():
            test_group, control_group = create_groups(anno, group_idx=i)
            stat, p = mannwhitneyu(np.array(test_group), np.array(control_group))  # , alternative='greater')
            print(f'P-value for U-test for class {id2desc[i]}: {p:.4E} [{len(test_group)}/{len(control_group)}]')
            print(f'Median (std): {np.median(test_group):.4f} ({np.std(test_group):.4f}) / '
                  f'{np.median(control_group):.4f} ({np.std(control_group):.4f})')
            plot_groups(test_group, control_group, path=f"/tmp/{id2desc[i]}.png")
    elif test == 'friedman-test':
        groups = create_pooled_groups(anno, id2desc)
        groups = list(groups.values())
        stat, p = friedmanchisquare(*groups)
        print(f'Test statistic and p-value for Friedmann test: {stat}, {p:.4E}, len: {[len(g) for g in groups]}')
    elif test == 'wilcoxon-test':
        groups = create_pooled_groups(anno, id2desc)
        for id in id2desc.keys():
            if id == 10:
                continue
            test_group = np.array(groups[id])
            control_group = np.array(groups[10])#np.average(np.array([v for k, v in groups.items() if k != id]), axis=0)
            stat, p = wilcoxon(test_group, control_group)  # , alternative='greater')
            print(f'P-value for U-test for class {id2desc[id]}: {p:.4E} [{len(test_group)}/{len(control_group)}]')


if __name__ == '__main__':
    run()
