from collections import namedtuple
from os.path import join

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, \
    cohen_kappa_score, roc_auc_score, average_precision_score

Score = namedtuple('Score', ['f1', 'precision', 'recall', 'accuracy', 'kappa', 'loss', 'roc', 'pr'])


class Scores:
    def __init__(self):
        self.columns = ['eye_id', 'label', 'prediction', 'probability', 'attention', 'position']
        self.data = pd.DataFrame(columns=self.columns)

    def add(self, preds: torch.Tensor, labels: torch.Tensor, tags: list = None, probs: torch.Tensor = None,
            attention: torch.Tensor = None, pos: torch.Tensor = None):
        if probs and probs.size(1) > 1:
            probs[:, 0] = probs[:, 1]
        new_data = tags if tags is not None else ['train' for i in range(len(labels.tolist()))], \
                   labels.tolist(), \
                   preds.tolist(), \
                   probs[:, 0].tolist() if probs is not None else [0 for i in range(len(labels.tolist()))], \
                   attention.tolist() if attention is not None else [0 for i in range(len(labels.tolist()))], \
                   pos.tolist() if pos is not None else [0 for i in range(len(labels.tolist()))]

        new_data_dict = {col: new_data[i] for i, col in enumerate(self.columns)}
        self.data = self.data.append(pd.DataFrame(new_data_dict), ignore_index=True)
        # pd.concat([self.data].extend(pd.DataFrame()), ignore_index=True)

    def calc_scores(self, as_dict: bool = False):
        # print(self.data['label'].tolist(), self.data['prediction'].tolist())
        score = calc_metrics(self.data['label'].tolist(), self.data['prediction'].tolist(),
                             self.data['probability'].tolist())
        if self.data['probability'].sum() != 0:
            print('Confusion matrix: \n ',
                  confusion_matrix(self.data['label'].tolist(), self.data['prediction'].tolist()))
        return score._asdict() if as_dict else score

    def calc_scores_eye(self, as_dict: bool = False, ratio: float = 0.5, top_percent: float = 1.0, mode: str = 'count'):
        """
        Calc scores for an eye if multiple pictures are available
        :param as_dict: Return normal dict, o/w named tuple
        :param ratio: percent threshold for an pos. prediction
        :param top_percent: percent of values considered
        :param mode: count / probs, consider binary predictions or their probabilites
        :return:
        """
        eye_data = pd.DataFrame(columns=self.columns)
        self.data.prediction = self.data.prediction.apply(lambda p: p[0] if type(p) == list else p)
        self.data.probability = self.data.probability.apply(lambda p: p[0] if type(p) == list else p)
        self.data.label = self.data.label.apply(lambda p: p[0] if type(p) == list else p)
        eye_groups = self.data.groupby('eye_id')  # create group for different eyes

        for name, group in eye_groups:
            num_voting_values = int(np.ceil(top_percent * len(group)))  # percentage of values considered
            if top_percent != 1.0:
                group.sort_values(by=['probability'], ascending=False)  # sort predictions by confidence

            pos = int(group['prediction'][:num_voting_values].sum())  # count positive predictions
            pos_probs = float(group['probability'][:num_voting_values].sum())
            eye_prediction = 1 if pos / num_voting_values >= ratio else 0
            eye_prediction_probs = pos_probs / num_voting_values

            eye_data = eye_data.append({
                self.columns[0]: name, self.columns[1]: group.iloc[0, 1], self.columns[2]: eye_prediction,
                self.columns[3]: pos_probs
            }, ignore_index=True)

        score = calc_metrics(eye_data['label'].tolist(), eye_data['prediction'].tolist(),
                             eye_data['probability'].tolist())
        return score._asdict() if as_dict else score

    def persist_scores(self, path, cur_epoch, scores):
        self.data.to_csv(join(path, f'{cur_epoch}_last_pad_model_{scores["f1"]:0.4}.csv'), index=False)
        self.data.to_csv(join(path, f'latest_pad_results.csv'), index=False)


def calc_metrics(labels: list, preds: list, probs: list, loss: float = 0.0) -> Score:
    # precision, recall, _ = precision_recall_curve(labels, probs)
    score = Score(f1_score(labels, preds),
                  precision_score(labels, preds),
                  recall_score(labels, preds),
                  accuracy_score(labels, preds),
                  cohen_kappa_score(labels, preds),
                  loss,
                  roc_auc_score(labels, probs),
                  average_precision_score(labels, probs))
    return score
