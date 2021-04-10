import pandas as pd
import transfer_learning
import multiple_instance_learning
from include.nn_utils import Score
from sklearn import metrics
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
    parser.add_argument('--strategy', '-s', help='MIL/CNN', type=str, default='CNN')

    args = parser.parse_args()
    
    score_df = pd.DataFrame()
    for i in range(args.folds):
        metric = transfer_learning.run(join(args.data, f'fold{i}'), args.model, args.epochs) if args.strategy == 'CNN' else multiple_instance_learning.run(join(args.data, f'fold{i}'), args.model, args.epochs)
        score_df = pd.concat([score_df, metric.data]).reset_index(drop=True)
    
    mean_score = {
            "f1": metrics.f1_score(score_df['label'].tolist(), score_df['prediction'].tolist()),
            "roc": metrics.roc_auc_score(score_df['label'].tolist(), score_df['probability'].tolist()),
            "pr": metrics.average_precision_score(score_df['label'].tolist(), score_df['probability'].tolist())
    }

    print(f'Avg scores for the PAD dataset (n={len(score_df)}): {mean_score}')
    sys.exit(0)



