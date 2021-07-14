import numpy as np
from matplotlib import pyplot as plt


def boxplot_groups(groups, id2desc, path='/tmp/boxplot_groupsizes.png', show_outliers=False):
    #fig = plt.figure()
    #bp = plt.boxplot(groups, showfliers=show_outliers)

    fig, ax = plt.subplots(figsize=(10, 7))
    if type(groups) == dict:
        ax.boxplot(groups.values(), showfliers=show_outliers)
    else:
        ax.boxplot(groups, showfliers=show_outliers)
    ax.set_xticklabels(id2desc.values())
    start, end = ax.get_ylim()
    #ax.yaxis.set_ticks(np.arange(0, end, 2))
    ax.grid(axis='y')

    #plt.xticks(list(id2desc.keys()), list(id2desc.values()))
    plt.savefig(path)
    plt.close(fig)


def plot_groups(test_group, control_group, path='tmp/output.png'):
    fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_axes([0, 0, 1, 1])
    bp = plt.boxplot([test_group, control_group], showfliers=False)
    plt.xticks([1, 2], ['test', 'control'])
    plt.savefig(path)
    plt.close(fig)


def plot_cooccurence_mat(mat, id2desc, out_path='/tmp/'):
    fig, ax = plt.subplots(figsize=(10, 7)) # plt.figure(figsize=(10, 7))
    plt.yticks(list(range(0, len(id2desc.values()))), id2desc.values())
    plt.xticks(list(range(0, len(id2desc.values()))), [t for t in id2desc.values()])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            text = ax.text(j, i, f'{mat[i, j]:.02f}', ha='center', va='center', color='w')
    plt.imshow(mat)
    plt.colorbar()
    plt.savefig(out_path + ('overlap.png'))
    plt.close(fig)