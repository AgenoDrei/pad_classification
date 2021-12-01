import argparse
import os
from os.path import join
from shutil import copy
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

DATA_FOLDER = ''

def run(input_path, labels_path, val_size):
    """
    Copy Paxos frames to a training and validation folder
    :param input_path: Absolute path to input folder
    :param labels_path: Absolute path to a label file
    :param val_size: Size of the validation set
    :return:
    """
    df = pd.read_csv(labels_path)
    df['ID'] = df['ID'].astype(str)
    df_val = pd.DataFrame(columns=df.columns)
    df_train = pd.DataFrame(columns=df.columns)

    X, y = df['ID'], df['diagnosis']
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size)
    split = next(splitter.split(X, y))

    for idx in tqdm(split[0], total=len(split[0]), desc='Train data'):  # idx of train set
        df_train = process_row(input_path, df.iloc[idx, 0], df.iloc[idx, 1], 'train', df_train)
    for idx in tqdm(split[1], total=len(split[1]), desc='Val data'):  # idx of validation set
        df_val = process_row(input_path, df.iloc[idx, 0], df.iloc[idx, 1], 'val', df_val)

    df_val.to_csv(join(input_path, 'labels_val.csv'), index=False)
    df_train.to_csv(join(input_path, 'labels_train.csv'), index=False)


def process_row(path, image, level, set, set_df):
    severity = 1 if level > 0 else 0
    set_df = set_df.append({'ID': image, 'diagnosis': level}, ignore_index=True)
    copy(join(path, DATA_FOLDER, image + '.jpg'), join(path, set, image + '.jpg'))
    return set_df


if __name__ == '__main__':
    a = argparse.ArgumentParser(description='Split Kaggle dataset into train and validation set using stratified shuffling')
    a.add_argument("--input", help="absolute path to input folder")
    a.add_argument("--labels", help="absolute path to the label file")
    a.add_argument("--valsize", help="Percentage of validation set size", type=float, default=0.1)
    args = a.parse_args()

    os.mkdir(join(args.input, 'train'))
    os.mkdir(join(args.input, 'val'))

    run(args.input, args.labels, args.valsize)






