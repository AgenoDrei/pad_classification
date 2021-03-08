import os
from collections import namedtuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, cohen_kappa_score, \
    precision_recall_curve, confusion_matrix, roc_auc_score, average_precision_score
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


def write_scores(writer, tag: str, scores: dict, cur_epoch: int, full_report: bool = False):
    writer.add_scalar(f'{tag}/f1', scores['f1'], cur_epoch)
    writer.add_scalar(f'{tag}/precision', scores['precision'], cur_epoch)
    writer.add_scalar(f'{tag}/recall', scores['recall'], cur_epoch)
    if scores.get('loss'): writer.add_scalar(f'{tag}/loss', scores['loss'], cur_epoch)
    if scores.get('roc'): writer.add_scalar(f'{tag}/roc', scores['roc'], cur_epoch)
    print(f'{tag[0].upper()}{tag[1:]} scores:\n F1: {scores["f1"]},\n Precision: {scores["precision"]},\n Recall: {scores["recall"]}')
    if full_report:
        print(f' Accuracy: {scores["accuracy"]},')
        print(f' ROC AUC: {scores["roc"]}')
        print(f' PR AUC: {scores["pr"]}')
        writer.add_scalar(f'{tag}/kappa', scores['kappa'], cur_epoch)
        writer.add_scalar(f'{tag}/accuracy', scores['accuracy'], cur_epoch)


Score = namedtuple('Score', ['f1', 'precision', 'recall', 'accuracy', 'kappa', 'loss', 'roc', 'pr'])


class Scores:
    def __init__(self):
        self.columns = ['eye_id', 'label', 'prediction', 'probability', 'attention', 'position']
        self.data = pd.DataFrame(columns=self.columns)

    def add(self, preds: torch.Tensor, labels: torch.Tensor, tags: list = None, probs: torch.Tensor = None, attention: torch.Tensor = None, pos: torch.Tensor = None):
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
        #print(self.data['label'].tolist(), self.data['prediction'].tolist())
        score = calc_metrics(self.data['label'].tolist(), self.data['prediction'].tolist(), self.data['probability'].tolist())
        if self.data['probability'].sum() != 0:
            print('Confusion matrix: \n ', confusion_matrix(self.data['label'].tolist(), self.data['prediction'].tolist()))
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

        score = calc_metrics(eye_data['label'].tolist(), eye_data['prediction'].tolist(), eye_data['probability'].tolist())
        return score._asdict() if as_dict else score


def calc_metrics(labels: list, preds: list, probs: list, loss: float = 0.0) -> Score:
    #precision, recall, _ = precision_recall_curve(labels, probs)
    score = Score(f1_score(labels, preds),
                  precision_score(labels, preds),
                  recall_score(labels, preds),
                  accuracy_score(labels, preds),
                  cohen_kappa_score(labels, preds),
                  loss,
                  roc_auc_score(labels, probs),
                  average_precision_score(labels, probs))
    return score
