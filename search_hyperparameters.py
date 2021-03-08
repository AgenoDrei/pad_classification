import copy
import numpy as np
import toml
import transfer_learning
import multiple_instance_learning
from include.nn_utils import Score
import argparse
import sys
from pprint import pprint as pp
from os.path import join
import math

CONFIG_PATH = 'config_hyperparameter_search.toml'
NUM_PERMUTATIONS = 20


def run(data_path, model_path, num_epochs, strategy, mode='random'):
    results = []
    config = toml.load(CONFIG_PATH)
    hp_space = config['hp']
    hp_permutations = get_hyperparameter_permutations(hp_space, mode)

    for hp in hp_permutations:
        scores, score_eyes = None, None
        if strategy == 'CNN':
            scores, score_eyes = transfer_learning.run(data_path, model_path, num_epochs)
        elif strategy == 'MIL':
            scores, score_eyes = multiple_instance_learning.run(data_path, model_path, num_epochs, custom_hp=hp)
        else:
            raise  Exception('Unknown learning strategy')

        results.append((hp, scores, score_eyes))
        print('Results for the hyperparameter permuation: ')
        pp(hp)
        print('Scores: ')
        pp(scores)
        print('Eye-Scores: ')
        pp(score_eyes)
        print('--------------------------')

    results = sorted(results, key=lambda d: d[1]['roc'])
    print('############ TOP 5 ############')
    for r in results[:5]:
        pp(r[0])
        print(f'Score, roc: {r[1]["roc"]}, f1: {r[1]["f1"]}, pr_auc: {r[1]["pr"]}')


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
    




