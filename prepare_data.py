import joblib as job
import argparse
import cv2
import os
import numpy as np
from os.path import join
import pandas as pd
import re
from pathlib import Path
from shutil import copy
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

COL_ID = 'ID'
COL_CLASS = 'pavk_FS_max'

def run(input_path, output_path, k, labels_path):
    df = pd.read_csv(labels_path)
    files = os.listdir(input_path)
    
    df[COL_ID] = df[COL_ID].astype(str)
    df[COL_ID] = df[COL_ID].str[:-1]
    df[COL_CLASS] = df[COL_CLASS].fillna(0)
    df.loc[df[COL_CLASS] > 4, COL_CLASS] = 4
    df[COL_CLASS] = df[COL_CLASS].astype(int)
    
    df_pat = df.groupby(COL_ID)[COL_CLASS, 'Eye'].agg(lambda d: ','.join([str(e) for e in d])).reset_index()
    df_pat[COL_CLASS] = df_pat[COL_CLASS].apply(lambda d: int(d[0]) if len(d) < 2 or d[0] == d[2] else -1) 
    
    splitter = StratifiedKFold(n_splits=k, shuffle=True)
    
    for i, split in enumerate(splitter.split(df_pat.drop([COL_CLASS], axis=1), df_pat[COL_CLASS])):
        print(f'Creating fold{i}...')
        df_train = df_pat.iloc[split[0]]
        df_val = df_pat.iloc[split[1]]

        for s, name in zip([df_train, df_val], ['train', 'val']):
            #s.to_csv(join(output_path, f'fold{i}', f'labels_{name}.csv'), index=False)
            out = pd.DataFrame(columns=[COL_ID, COL_CLASS])
            for row in s.itertuples():
                out = out.append({COL_ID: row[1] + row[3][0], COL_CLASS: row[2]}, ignore_index=True)
                [copy(join(input_path, f), join(output_path, f'fold{i}', name, f)) for f in files if row[1] + row[3][0] in f]
                if len(row[3]) > 1: # Two eyes exist for patient
                    out = out.append({COL_ID: row[1] + row[3][2], COL_CLASS: row[2]}, ignore_index=True)
                    [copy(join(input_path, f), join(output_path, f'fold{i}', name, f)) for f in files if row[1] + row[3][2] in f]
            out.to_csv(join(output_path, f'fold{i}', f'labels_{name}.csv'), index=False)


if __name__ == '__main__':
    a = argparse.ArgumentParser(description='Split Paxos trainings data into train/val set using stratified shuffling')
    a.add_argument("--input", help="absolute path to input folder")
    a.add_argument("--output", help="absolute path to output folder")
    a.add_argument("--labels", help="absolute path to input folder")
    a.add_argument("--folds", "-k", help="Number of folds", type=int, default=10)
    args = a.parse_args()
    
    os.mkdir(args.output)
    for i in range(args.folds):
        os.mkdir(join(args.output, f'fold{i}'))
        os.mkdir(join(args.output, f'fold{i}', 'train'))
        os.mkdir(join(args.output, f'fold{i}', 'val'))

    run(args.input, args.output, args.folds, args.labels)







