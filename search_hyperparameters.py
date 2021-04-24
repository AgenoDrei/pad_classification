import copy
import numpy as np
import toml
import transfer_learning
import multiple_instance_learning
from include.nn_metrics import Score
import argparse
import sys
from pprint import pprint as pp
from os.path import join
import time
import os


CONFIG_PATH = 'config_hyperparameter_search.toml'
NUM_PERMUTATIONS = 40


def run(data_path, model_path, num_epochs, strategy, mode='random', num_results=5):
    results = []
    config = toml.load(CONFIG_PATH)
    hp_space = config['hp']
    hp_permutations = get_hyperparameter_permutations(hp_space, mode)

    working_path = f'{time.strftime("%Y%m%d_%H%M")}_pad_hp_search/'
    os.mkdir(working_path)

    for hp in hp_permutations:
        metric = None
        if strategy == 'CNN':
            metric = transfer_learning.run(data_path, model_path, num_epochs)
        elif strategy == 'MIL':
            metric = multiple_instance_learning.run(data_path, model_path, num_epochs, custom_hp=hp)
        else:
            raise Exception('Unknown learning strategy')

        results.append((hp, metric.calc_scores(as_dict=True), metric.calc_scores_eye(as_dict=True)))
        print('Results for the hyperparameter permuation: ')
        pp(hp)
        print('Scores: ')
        pp(metric.calc_scores(as_dict=True))
        print('Eye-Scores: ')
        pp(metric.calc_scores_eye(as_dict=True))
        print('--------------------------')

    results = sorted(results, key=lambda d: d[1]['roc'], reverse=True)
    print('############ TOP 5 ############')
    with open('hp_search_results.txt', 'a') as out:
        for r in results[:num_results]:
            pp(r[0])
            print(f'Score, roc: {r[1]["roc"]}, f1: {r[1]["f1"]}, pr_auc: {r[1]["pr"]}')
            pp(r[0], stream=out)
            print(f'Score, roc: {r[1]["roc"]}, f1: {r[1]["f1"]}, pr_auc: {r[1]["pr"]}', file=out)


def get_hyperparameter_permutations(hp_space, mode):
    hp_permutations = []
    relevant_hp_keys = []
    for k, v in hp_space.items():
        if type(v) == list:
            relevant_hp_keys.append(k)

    print(f'There a {sum([len(hp_space[k]) for k in relevant_hp_keys])} possible combinations '
          f'of the given hyperparameter values')

    if mode == 'random':
        for i in range(NUM_PERMUTATIONS):
            hps = copy.deepcopy(hp_space)
            for key in relevant_hp_keys:
                possible_values = hps[key]
                hps[key] = possible_values[np.random.randint(0, len(possible_values))]
            hp_permutations.append(hps)
    elif mode == 'exhaustive':
        raise Exception(f'Not yet implemented')
    else:
        raise Exception(f'Mode {mode} not available')

    print(hp_permutations)
    return hp_permutations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PAD k-fold')
    parser.add_argument('--data', help='Path for training data', type=str)
    parser.add_argument('--epochs', help='Number of training epochs', type=int, default=50)
    parser.add_argument('--model', help='Path to pretrained model', type=str, default=None)
    parser.add_argument('--strategy', '-s', help='MIL/CNN', type=str, default='CNN')
    args = parser.parse_args()

    run(args.data, args.model, args.epochs, args.strategy)
    sys.exit(0)
    




