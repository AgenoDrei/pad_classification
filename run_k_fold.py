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
    mean_score = Score(0, 0, 0, 0, 0, 0, 0, 0)._asdict()
    mean_score_eyes = Score(0, 0, 0, 0, 0, 0, 0, 0)._asdict()
    for i in range(args.folds):
        scores, score_eyes = run(join(args.data, f'fold{i}'), args.model, args.epochs)
        mean_score = {k: mean_score[k] + v / args.folds for k, v in scores.items()}
        mean_score_eyes = {k: mean_score_eyes[k] + v / args.folds for k, v in score_eyes.items()}

    #std = math.sqrt(sum([(f - avg_f1)**2 / (len(f1_list)-1) for f in f1_list]))
    print(f'Avg scores for the PAD dataset: {mean_score}')
    print(f'Avg eye scores for the PAD dataset: {mean_score_eyes}')
    #print('Standard divation for PAD dataset: ', std)
    sys.exit(0)



