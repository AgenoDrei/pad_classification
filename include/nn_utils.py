import os
from collections import namedtuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, cohen_kappa_score, \
    precision_recall_curve
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils


def display_examples(ds):
    fig = plt.figure(figsize=(10, 10))

    for i in range(0, 40, 10):
        sample = ds[i]
        ax = plt.subplot(1, 4, i // 10 + 1)
        plt.tight_layout()
        ax.set_title(f'Sample #{i}- {sample["label"]}')
        ax.axis('off')
        plt.imshow(sample['image'])

    plt.show()


# Helper function to show a batch
def show_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, label_batch = sample_batched['image'], sample_batched['label']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.show()


def save_batch(batch, path):
    images_batch, label_batch = batch['image'], batch['label']
    for i, img in enumerate(images_batch):
        cv2.imwrite(os.path.join(path, f'{i}_{label_batch[i]}.png'), img.numpy().transpose((1, 2, 0)))


def get_video_desc(video_path, only_eye=False):
    """
    Get video description in easy usable dictionary
    :param video_path: path / name of the video_frame file
    :param only_eye: Only returns the first part of the string
    :return: dict(eye_id, snippet_id, frame_id, confidence), only first two are required
    """
    video_name = os.path.basename(video_path)
    video_name = os.path.splitext(video_name)[0]
    info_parts = video_name.split("_")

    if len(info_parts) == 1 or only_eye:
        return {'eye_id': info_parts[0]}
    elif len(info_parts) == 2:
        return {'eye_id': info_parts[0], 'snippet_id': int(info_parts[1])}
    elif len(info_parts) > 3:
        return {'eye_id': info_parts[0], 'snippet_id': int(info_parts[1]), 'frame_id': int(info_parts[3]),
                'confidence': info_parts[2]}
    else:
        return {'eye_id': ''}


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def calc_scores_from_confusion_matrix(cm):
    """
    Calc precision, recall and f1 score from a 2x2 numpy confusion matrix
    :param cm: confusion matrix, numpy 2x2 ndarray
    :return: dict(precision, recall, f1)
    """
    tp = cm[1, 1].item()
    fp = cm[0, 1].item()
    fn = cm[1, 0].item()

    if tp + fp == 0:
        return {'precision': 0, 'recall': 0, 'f1': 0}

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * tp) / (2 * tp + fp + fn)

    return {'precision': precision, 'recall': recall, 'f1': f1}


def write_f1_curve(majority_dict, writer):
    roc_data = majority_dict.get_roc_data()
    roc_scores = {}
    for i, d in enumerate(roc_data.values()):
        roc_scores[i] = f1_score(d['labels'], d['predictions'])
    print('F1 Curve: ', end='')
    for key, val in roc_scores.items():
        writer.add_scalar('val/f1_roc', val, key)
        print(f'{key}: {val}', end=', ')


def write_pr_curve(majority_dict, writer: SummaryWriter):
    probs, labels, names = majority_dict.get_probabilities_and_labels().values()
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    print('PR Curve (Recall: Precision): ')
    for p, r in zip(precision, recall):
        print(f' {r}: {p}')

    # writer.add_pr_curve('eval/pr', labels, probs, 0)
    # fig = plt.figure()
    # plt.plot(recall, precision)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # writer.add_figure('eval/pr', fig)


def write_scores(writer, tag: str, scores: dict, cur_epoch: int, full_report: bool = False):
    writer.add_scalar(f'{tag}/f1', scores['f1'], cur_epoch)
    writer.add_scalar(f'{tag}/precision', scores['precision'], cur_epoch)
    writer.add_scalar(f'{tag}/recall', scores['recall'], cur_epoch)
    if scores.get('loss'): writer.add_scalar(f'{tag}/loss', scores['loss'], cur_epoch)
    if full_report:
        writer.add_scalar(f'{tag}/kappa', scores['kappa'], cur_epoch)
        writer.add_scalar(f'{tag}/accuracy', scores['accuracy'], cur_epoch)
    print(
        f'{tag[0].upper()}{tag[1:]} scores:\n F1: {scores["f1"]},\n Precision: {scores["precision"]},\n Recall: {scores["recall"]}')


class MajorityDict:
    def __init__(self):
        self.dict = {}

    def add(self, predictions, ground_truth, key_list, probabilities=None):
        """
        Add network predictions to Majority Dict, has to be called for every batch of the validation set
        :param probabilities:
        :param predictions: list of predictions
        :param ground_truth: list of correct, known labels
        :param key_list: list of keys (like video id)
        :return: None
        """
        for i, (true, pred) in enumerate(zip(ground_truth, predictions)):
            if self.dict.get(key_list[i]):
                entry = self.dict[key_list[i]]
                entry[str(pred)] += 1
                entry['count'] += 1
            else:
                self.dict[key_list[i]] = {'0': 0 if int(pred) else 1, '1': 1 if int(pred) else 0, 'count': 1,
                                          'label': int(true)}

        if probabilities:
            for gt, pred, prob, name in zip(ground_truth, predictions, probabilities, key_list):
                if self.dict.get(name):
                    entry = self.dict.get(name)
                    entry['results'][round(prob, 8)] = pred
                    entry['count'] += 1
                else:
                    self.dict[name] = {'label': int(gt), 'count': 1, 'results': {}}
                    self.dict[name]['results'][round(prob, 8)] = pred

    def get_probabilities_and_labels(self):
        labels, probs, names = [], [], []
        for i, item in self.dict.items():
            probs.append(item['1'] / (item['1'] + item['0']))
            labels.append(item['label'])
            names.append(i)
        return {'probabilities': probs, 'labels': labels, 'names': names}

    def get_predictions_and_labels(self, ratio: float = 0.5):
        """
        Do majority voting to get final predicions aggregated over all elements sharing the same key
        :param ratio: Used to shange majority percentage (default 50/50)
        :return: dict(predictions, labels, names)
        """
        labels, preds, names = [], [], []
        for i, item in self.dict.items():
            if item['1'] > ratio * (item['0'] + item['1']):
                preds.append(1)
            else:
                preds.append(0)
            labels.append(item['label'])
            names.append(i)
        return {'predictions': preds, 'labels': labels, 'names': names}

    def get_eye_predictions_from_best_scores(self, num_best_scores: int = 10, ratio: float = 0.5):
        labels, preds, names = [], [], []
        for i, item in self.dict.items():
            labels.append(item['label'])
            names.append(i)
            sorted_res = sorted(item['results'].items(), reverse=True)[:num_best_scores]
            sums = (0, 0)
            for prob, pred in sorted_res:
                sums[int(pred)] += prob
            if sums[1] > ratio * (sums[0] * sums[1]):
                preds.append(1)
            else:
                preds.append(0)
        return {'predictions': preds, 'labels': labels, 'names': names}

    def get_roc_data(self, step_size: float = 0.05):
        """
        Generate predictions for different thresholds
        :param step_size: step_size between different thresholds
        :return: dict(step: dict)
        """
        roc_data = {}
        for i in np.arange(0, 1.001, step_size):
            roc_data[i] = self.get_predictions_and_labels(ratio=i)
        return roc_data


Score = namedtuple('Score', ['f1', 'precision', 'recall', 'accuracy', 'kappa', 'loss'])


class Scores:
    def __init__(self):
        self.columns = ['eye_id', 'label', 'prediction', 'probability']
        self.data = pd.DataFrame(columns=self.columns)

    def add(self, preds: torch.Tensor, labels: torch.Tensor, tags: list = None, probs: torch.Tensor = None):
        new_data = tags if tags is not None else ['train' for i in range(len(labels.tolist()))], \
                   labels.tolist(), \
                   preds.tolist(), \
                   probs.tolist() if probs is not None else [0 for i in range(len(labels.tolist()))]

        new_data_dict = {col: new_data[i] for i, col in enumerate(self.columns)}
        self.data = self.data.append(pd.DataFrame(new_data_dict), ignore_index=True)
        # pd.concat([self.data].extend(pd.DataFrame()), ignore_index=True)

    def calc_scores(self, as_dict: bool = False):
        print(self.data['label'].tolist(), self.data['prediction'].tolist())
        score = Score(f1_score(self.data['label'].tolist(), self.data['prediction'].tolist()),
                      precision_score(self.data['label'].tolist(), self.data['prediction'].tolist()),
                      recall_score(self.data['label'].tolist(), self.data['prediction'].tolist()),
                      accuracy_score(self.data['label'].tolist(), self.data['prediction'].tolist()),
                      cohen_kappa_score(self.data['label'].tolist(), self.data['prediction'].tolist()), 0)
        return score._asdict() if as_dict else score

    def calc_scores_eye(self, as_dict: bool = False, ratio: float = 0.5, top_percent=1.0):
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
            eye_prediction = 1 if pos / num_voting_values >= ratio else 0
            eye_data = eye_data.append({
                self.columns[0]: name, self.columns[1]: group.iloc[0, 1], self.columns[2]: eye_prediction,
                self.columns[3]: 0.0
            }, ignore_index=True)

        score = Score(f1_score(eye_data['label'].tolist(), eye_data['prediction'].tolist()),
                      precision_score(eye_data['label'].tolist(), eye_data['prediction'].tolist()),
                      recall_score(eye_data['label'].tolist(), eye_data['prediction'].tolist()),
                      accuracy_score(eye_data['label'].tolist(), eye_data['prediction'].tolist()),
                      cohen_kappa_score(eye_data['label'].tolist(), eye_data['prediction'].tolist()), 0)
        return score._asdict() if as_dict else score
