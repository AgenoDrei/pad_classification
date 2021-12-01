import os
import cv2
import click
import pickle
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, mannwhitneyu, wilcoxon
from scikit_posthocs import posthoc_nemenyi_friedman
from tqdm import tqdm

from include.plotting import plot_groups, plot_cooccurence_mat, boxplot_groups


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
    if os.path.exists(f'/tmp/annos{6 if combine else 10}.pkl'):
        file = open(f'/tmp/annos{6 if combine else 10}.pkl', 'rb')
        anno = pickle.load(file)
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

    output_file = open(f'/tmp/annos{6 if combine else 10}.pkl', 'wb')
    pickle.dump(annotations, output_file)
    output_file.close()
    return annotations


def create_groups(anno, group_idx=0, only_empty=True):
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


def create_vessel_groups(anno):
    artery_group, vein_group = [], []
    artery_ids, vein_ids = [0, 4, 2, 6][:], [1, 5, 3, 7][:]
    for sample in anno.values():
        w_agg = {'artery': [0.0, 0], 'vein': [0.0, 0]}
        # w_agg = {'artery': [], 'vein': []}
        for cell in sample:
            if len(cell['visible_classes']) > 1:
                continue
            for aid in artery_ids:
                if aid in cell['visible_classes']:
                    # w_agg['artery'].append(cell['weight'])
                    w_agg['artery'][0] += cell['weight']
                    w_agg['artery'][1] += 1
            for vid in vein_ids:
                if vid in cell['visible_classes']:
                    # w_agg['vein'].append(cell['weight'])
                    w_agg['vein'][0] += cell['weight']
                    w_agg['vein'][1] += 1
        artery_group.append(w_agg['artery'][0] / (w_agg['artery'][1] + 10e-100)) # artery_group.append(np.median(w_agg['artery']))
        vein_group.append(w_agg['vein'][0] / (w_agg['vein'][1] + 10e-100)) # vein_group.append(np.median(w_agg['vein']))
    print(f'Arteries mean (std): {np.mean(artery_group):.4f} ({np.std(artery_group):.4f})')
    print(f'Veins mean (std): {np.median(vein_group):.4f} ({np.std(vein_group):.4f})')
    return artery_group, vein_group


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


def create_overlap_matrix(annotations, id2desc, path='/tmp/', normalize=False):
    trans = {0: 0, 2: 1, 4: 2, 6: 3, 8: 4, 9: 5}
    ol_mat = np.zeros((len(id2desc.values()), len(id2desc.values())))
    for sample in annotations.values():
        for cell in sample:
            classes = cell['visible_classes']
            if len(classes) == 1:  # if the patch shows only ONE class
                if len(id2desc.values()) == 6:  # if artery / vein classes are combined
                    ol_mat[trans[classes[0]], trans[classes[0]]] += 1
                else:
                    ol_mat[classes[0], classes[0]] += 1
            elif len(classes) > 1:  # more than 1 class
                classes = sorted(list(set(classes)))
                for a in classes:
                    if len(id2desc.values()) == 6:
                        ol_mat[trans[a], trans[a]] += 1
                    else:
                        ol_mat[a, a] += 1
                for a, b in list(itertools.combinations(classes, 2)):
                    if len(id2desc.values()) == 6:
                        ol_mat[trans[a], trans[b]] += 1
                    else:
                        ol_mat[a, b] += 1

    if normalize:
        diag = np.diagonal(ol_mat)
        row_sums = ol_mat.sum(axis=1, keepdims=True)
        ol_mat = ol_mat / diag

    ol_mat = ol_mat + ol_mat.T - np.diag(ol_mat.diagonal())
    plot_cooccurence_mat(ol_mat, id2desc, out_path=path)
    return ol_mat


def create_groupsize_plots(anno, id2desc):
    id2desc[10] = 'no_class'
    groups = {k: [] for k in id2desc.keys()}
    for eye in anno.values():
        count = {k: 0 for k in id2desc.keys()}
        for cell in eye:
            if len(cell['visible_classes']) == 0:
                count[10] += 1
            for v in cell['visible_classes']:
                count[v] += 1
        for k in groups.keys():
            groups[k].append(count[k])
    boxplot_groups(groups, id2desc)
    return groups


def create_weight_by_class_plots(anno, id2desc):
    groups = [create_groups(anno, gid, only_empty=True)[0] for gid in id2desc.keys()][:10]
    groups.append(create_groups(anno, 0, only_empty=True)[1])
    boxplot_groups(groups, id2desc, path='/tmp/weight_by_class_boxplot.png', show_outliers=False)


@click.command()
@click.option('--attention_weights', '-w', help='Path to MIL results containing weights for images')
@click.option('--annotations', '-a', help='Path to VIA annotation file')
@click.option('--data', '-d', help='Path to the image dataset folder')
@click.option('--test', help='Type of statistical test used (u-test / friedman-test / wilcoxon-test / nemenyi-test)',
              default='u-test')
@click.option('--combine/--separate', help='Combine vessels into one class', default=False)
@click.option('--normalize/--count', help='Normalize co-occurence', default=False)
def run(attention_weights, annotations, data, test='u-test', combine=False, normalize=False):
    df = pd.read_csv(attention_weights)

    file = open(annotations, 'rb')
    anno = pickle.load(file)
    file.close()

    id2desc = {0: "STA", 2: "SNA", 4: "ITA", 6: "INA", 8: "MAC", 9: "OD"} if combine \
        else {0: "STA", 1: "STV", 2: "SNA", 3: "SNV", 4: "ITA", 5: "ITV", 6: "INA", 7: "INV", 8: "MAC", 9: "OD"}
    anno = enrich_annotations(anno, df, data, combine=combine)
    create_overlap_matrix(anno, id2desc, normalize=normalize)
    create_groupsize_plots(anno, id2desc)
    create_weight_by_class_plots(anno, id2desc)

    if test == 'u-test':
        for i in id2desc.keys():
            if i == 10:
                continue
            test_group, control_group = create_groups(anno, group_idx=i)
            stat, p = mannwhitneyu(np.array(test_group), np.array(control_group))  # , alternative='greater')
            print(f'P-value for U-test for class {id2desc[i]}: {p:.4E} [{len(test_group)}/{len(control_group)}]')
            print(f'Median (std): {np.median(test_group):.4f} ({np.std(test_group):.4f}) / '
                  f'{np.median(control_group):.4f} ({np.std(control_group):.4f})')
    elif test == 'friedman-test':
        groups = create_pooled_groups(anno, id2desc)
        groups = list(groups.values())
        stat, p = friedmanchisquare(*groups[:-1])
        print(f'Test statistic and p-value for Friedmann test: {stat}, {p:.4E}, len: {[len(g) for g in groups]}')
    elif test == 'wilcoxon-test':
        groups = create_pooled_groups(anno, id2desc)
        for cls_id in id2desc.keys():
            if cls_id == 10:
                continue
            test_group = np.array(groups[cls_id])
            control_group = np.array(groups[
                                         10])  # np.average(np.array([v for k, v in groups.items() if k != id]), axis=0)  # np.array(groups[10])
            stat, p = wilcoxon(test_group, control_group)  # , alternative='greater')
            print(
                f'P-value for wilcoxon-test for class {id2desc[cls_id]}: {p:.4E} [{len(test_group)}/{len(control_group)}]')
    elif test == 'vessel-test':
        artery_group, vein_group = create_vessel_groups(anno)
        stat, p = wilcoxon(np.array(artery_group), np.array(vein_group), alternative='less')
        print(f'P-value for wilcoxon-test for vessel types: {p:.4E}, stat: {stat}')
    elif test == 'nemenyi-test':
        groups = create_pooled_groups(anno, id2desc)
        print(groups)
        groups = list(groups.values())
        stat = posthoc_nemenyi_friedman(np.array(groups).T)
        print(stat)


if __name__ == '__main__':
    run()
