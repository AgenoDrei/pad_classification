from os.path import join
import argparse
import os
import sys
import time
import toml
import torch
import torch.optim as optim
from pretrainedmodels import inceptionv4
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm
from typing import Tuple

from include.nn_datasets import RetinaDataset, SegmentsDataset, get_validation_pipeline, get_training_pipeline, \
    RetinaBagDataset, get_dataset
from include.nn_models import BagNet
from include.nn_utils import dfs_freeze, Scores, write_scores, Score

RES_PATH = ''


def run(base_path, model_path, num_epochs):
    setup_log(base_path)
    config = toml.load('config_mil.toml')
    hp = config['hp']
    hp['pretraining'] = True if model_path else False
    print('--------Configuration---------- \n ', config)

    device = torch.device(config['gpu_name'] if torch.cuda.is_available() else "cpu")
    print(f'Working on {base_path}!')
    print(f'using device {device}')

    aug_pipeline_train = get_training_pipeline(0, hp['crop_size'], mode='mil', strength=hp['aug_strength'])
    aug_pipeline_val = get_validation_pipeline(0, hp['crop_size'], mode='mil')

    loaders = get_dataset(RetinaBagDataset, base_path, hp, aug_pipeline_train, aug_pipeline_val, config['num_workers'])
    net = prepare_model(model_path, hp, device)
    optimizer_ft = optim.Adam([{'params': net.feature_extractor_part1.parameters(), 'lr': 1e-5},
                               {'params': net.feature_extractor_part2.parameters()},  # , 'lr': 1e-5},
                               {'params': net.attention.parameters()},
                               {'params': net.att_v.parameters()},
                               {'params': net.att_u.parameters()},
                               {'params': net.att_weights.parameters()},
                               {'params': net.classifier.parameters()}], lr=hp['learning_rate'],
                              weight_decay=hp['weight_decay'])

    criterion = CrossEntropyLoss()
    plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=12, verbose=True)

    desc = f'_transfer_pad_{str("_".join([k[0] + str(hp) for k, hp in hp.items()]))}'
    writer = SummaryWriter(comment=desc)
    best_model, scores, eye_scores = train_model(net, criterion, optimizer_ft, plateau_scheduler, loaders, device,
                                                 writer, num_epochs=num_epochs)
    return scores, eye_scores


def setup_log(data_path):
    global RES_PATH
    RES_PATH = f'{time.strftime("%Y%m%d_%H%M")}_{os.path.basename(data_path)}_PAD/'
    os.mkdir(RES_PATH)


def prepare_model(model_path, hp, device):
    net = None
    if hp['network'] == 'AlexNet':
        net = models.alexnet(pretrained=True)
        net.classifier[-1] = Linear(net.classifier[-1].in_features, 2)

    if hp['pretraining']:
        net.load_state_dict(torch.load(model_path, map_location=device))
        print('Loaded stump: ', len(net.features))
    net = BagNet(net, num_attention_neurons=hp['attention_neurons'], attention_strategy=hp['attention'],
                 pooling_strategy=hp['pooling'], stump_type=hp['network'])
    print(f'Model info: {net.__class__.__name__}, #frozen layer: {hp["freeze"]}')
    return net.to(device)


def train_model(model, criterion, optimizer, scheduler, loaders, device, writer, num_epochs=50):
    since = time.time()
    best_f1_val = -1
    model.to(device)
    val_scores, val_eye_scores = {}, {}

    for epoch in range(num_epochs):
        print(f'{time.strftime("%H:%M:%S")}> Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        running_loss = 0.0
        metrics = Scores()
        # Iterate over data.
        for i, batch in tqdm(enumerate(loaders[0]), total=len(loaders[0]), desc=f'Epoch {epoch}'):
            inputs = batch['frames'].to(device, dtype=torch.float)
            labels = batch['label'].to(device)

            model.train()
            optimizer.zero_grad()
            loss, _, _ = model.calculate_objective(inputs, labels)
            error, pred = model.calculate_classification_error(inputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            metrics.add(pred, labels)

        train_scores = metrics.calc_scores(as_dict=True)
        train_scores['loss'] = running_loss / len(loaders[0].dataset)
        write_scores(writer, 'train', train_scores, epoch)
        val_loss, val_scores, val_eye_scores = validate(model, criterion, loaders[1], device, writer, epoch)

        best_f1_val = val_scores['f1'] if val_scores['f1'] > best_f1_val else best_f1_val

        scheduler.step(val_loss)

    time_elapsed = time.time() - since
    print(f'{time.strftime("%H:%M:%S")}> Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    torch.save(model.state_dict(), join(RES_PATH, f'model_pad_mil.pth'))
    return model, val_scores, val_eye_scores


def validate(model, criterion, loader, device, writer, cur_epoch, calc_roc=False) -> Tuple[float, dict, dict]:
    model.eval()
    running_loss = 0.0
    perf_metrics = Scores()

    for i, batch in tqdm(enumerate(loader), total=len(loader), desc='Validation'):
        inputs = batch['frames'].to(device, dtype=torch.float)
        labels = batch['label'].to(device)
        names = batch['name']

        with torch.no_grad():
            loss, attention_weights, probs = model.calculate_objective(inputs, labels)
            error, preds = model.calculate_classification_error(inputs, labels)
            running_loss += loss.item()

        perf_metrics.add(preds, labels, probs=probs, tags=names)

    scores = perf_metrics.calc_scores(as_dict=True)
    scores['loss'] = running_loss / len(loader.dataset)
    # scores_eye = perf_metrics.calc_scores_eye(as_dict=True, ratio=0.5)
    # write_scores(writer, 'eye_val', scores_eye, cur_epoch, full_report=True)
    write_scores(writer, 'val', scores, cur_epoch, full_report=True)
    perf_metrics.data.to_csv(join(RES_PATH, f'{cur_epoch}_last_pad_model_{scores["f1"]:0.3}.csv'), index=False)

    return running_loss / len(loader.dataset), scores, Score(0, 0, 0, 0, 0, 0, 0, 0)._asdict()


if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using torch with GPU {torch.cuda.is_available()}')

    parser = argparse.ArgumentParser(description='Multiple instance learning cause once is not enough')
    parser.add_argument('--data', help='Path for training data', type=str)
    parser.add_argument('--epochs', help='Number of training epochs', type=int, default=50)
    parser.add_argument('--model', help='Path to pretrained model', type=str, default=None)
    args = parser.parse_args()

    run(args.data, args.model, args.epochs)
    sys.exit(0)
