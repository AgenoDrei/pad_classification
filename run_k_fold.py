import os
from pprint import pprint as pp

import pandas as pd
import transfer_learning
import multiple_instance_learning
from include.nn_metrics import Score
from sklearn import metrics
import argparse
import sys
from os.path import join
import time
from include.nn_report import Reporting


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PAD k-fold')
    parser.add_argument('--data', help='Path for training data', type=str)
    parser.add_argument('--epochs', help='Number of training epochs', type=int, default=50)
    parser.add_argument('--model', help='Path to pretrained model', type=str, default=None)
    parser.add_argument('--folds', '-k', help='number of folds', type=int)
    parser.add_argument('--strategy', '-s', help='MIL/CNN', type=str, default='CNN')

    args = parser.parse_args()
    working_path = f'{time.strftime("%Y%m%d_%H%M")}_pad_kfolds/'
    os.mkdir(working_path)
    
    score_df = pd.DataFrame()
    for i in range(args.folds):
        writer = Reporting(log_dir=f'{working_path}fold{i}')
        metric = None
        if args.strategy == 'CNN':
            transfer_learning.RES_PATH = working_path
            metric = transfer_learning.run(join(args.data, f'fold{i}'), args.model, args.epochs, custom_writer=writer)
        elif args.strategy == 'MIL':
            multiple_instance_learning.RES_PATH = working_path
            metric = multiple_instance_learning.run(join(args.data, f'fold{i}'), args.model, args.epochs, custom_writer=writer)

        score_df = pd.concat([score_df, metric.data]).reset_index(drop=True)
    
    mean_score = {
            "f1": metrics.f1_score(score_df['label'].tolist(), score_df['prediction'].tolist()),
            "precision": metrics.precision_score(score_df['label'].tolist(), score_df['prediction'].tolist()),
            "recall": metrics.recall_score(score_df['label'].tolist(), score_df['prediction'].tolist()),
            "acc": metrics.accuracy_score(score_df['label'].tolist(), score_df['prediction'].tolist()),
            "roc": metrics.roc_auc_score(score_df['label'].tolist(), score_df['probability'].tolist()),
            "pr": metrics.average_precision_score(score_df['label'].tolist(), score_df['probability'].tolist())
    }

    print(f'Avg scores for the PAD dataset (n={len(score_df)}): {mean_score}')
    with open(os.path.join(working_path, 'kfold_results.txt'), 'a') as out:
        pp(mean_score, stream=out)

    sys.exit(0)



