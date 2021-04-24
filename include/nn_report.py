import time
from typing import Dict

import pandas as pd
from torch.utils.tensorboard import SummaryWriter


def print_scores(scores, tag):
    print(f'{tag[0].upper()}{tag[1:]} scores:')
    print(f' F1: {scores["f1"]}')
    print(f' ROC AUC: {scores["roc"]}')
    print(f' PR AUC: {scores["pr"]}')
    print(f' Accuracy: {scores["accuracy"]},')
    print(f' Precision: {scores["precision"]}')
    print(f' Recall: {scores["recall"]}')


class Reporting:
    def __init__(self, log_dir: str = None, writer_desc: str=f'{time.strftime("%Y%m%d_%H%M_report")}'):
        self.writer = SummaryWriter(log_dir=log_dir, comment=writer_desc)
        self.history_cols = ['epoch', 'f1', 'roc']
        self.history = {
            'val': pd.DataFrame(columns=self.history_cols),
            'val_eye': pd.DataFrame(columns=self.history_cols)
        }

    def write_scores(self, tag: str, scores: dict, cur_epoch: int):
        self.writer.add_scalar(f'{tag}/f1', scores['f1'], cur_epoch)
        self.writer.add_scalar(f'{tag}/precision', scores['precision'], cur_epoch)
        self.writer.add_scalar(f'{tag}/recall', scores['recall'], cur_epoch)
        if scores.get('loss'):
            self.writer.add_scalar(f'{tag}/loss', scores['loss'], cur_epoch)
        if scores.get('roc'):
            self.writer.add_scalar(f'{tag}/roc', scores['roc'], cur_epoch)
        if scores.get('kappa'):
            self.writer.add_scalar(f'{tag}/kappa', scores['kappa'], cur_epoch)
        if scores.get('accuracy'):
            self.writer.add_scalar(f'{tag}/accuracy', scores['accuracy'], cur_epoch)

        print_scores(scores, tag)
        if self.history.get(tag):
            line = {
                self.history_cols[0]: cur_epoch,
                self.history_cols[1]: scores['f1'],
                self.history_cols[2]: scores['roc']
            }
            self.history['tag'] = self.history['tag'].append(line, ignore_index=True)

    def write_hyperparameter(self, tag: str, hps: Dict):
        if self.history.get(tag) is None:
            raise Exception(f'Invalid metric set selected: {tag}')
        self.writer.add_hparams(hps, self.history[tag].tail(1).to_dict('records')[0])

    def __del__(self):
        self.writer.close()
