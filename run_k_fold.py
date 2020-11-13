from transfer_learning import run
from include.nn_utils import Score
import argparse
import sys
from os.path import join
import math

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PAD k-fold')
    parser.add_argument('--data', help='Path for training data', type=str)
    parser.add_argument('--epochs', help='Number of training epochs', type=int, default=50)
    parser.add_argument('--model', help='Path to pretrained model', type=str, default=None)
    parser.add_argument('--folds', '-k', help='number of folds', type=int)

    args = parser.parse_args()
    
    avg_f1, avg_roc = 0, 0
    f1_list, roc_list = [], []
    mean_score = {'f1': 0, 'roc': 0, 'pr': 0}
    for i in range(args.folds):
        scores = run(join(args.data, f'fold{i}'), args.model, args.epochs)
        mean_score['f1'] += scores['f1'] / args.folds
        mean_score['roc'] += scores['roc'] / args.folds
        mean_score['pr'] += scores['pr'] / args.folds

    #std = math.sqrt(sum([(f - avg_f1)**2 / (len(f1_list)-1) for f in f1_list]))
    print(f'Avg scores for the PAD dataset: {mean_score}')
    #print('Standard divation for PAD dataset: ', std)
    sys.exit(0)



