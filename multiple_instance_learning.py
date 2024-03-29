import argparse
import os
import shutil
import sys
import time
from os.path import join
from typing import Tuple
import toml
import torch
import torch.optim as optim
from torch import nn
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import lr_scheduler
from torchvision import models
from tqdm import tqdm
from include.nn_datasets import get_validation_pipeline, get_training_pipeline, \
    RetinaBagDataset, get_dataset
from include.nn_models import BagNet
from include.nn_report import Reporting
from include.nn_metrics import Score, Scores

RES_PATH = ''


def run(base_path, model_path, num_epochs, custom_hp=None, custom_writer=None):
    setup_log(base_path)
    config = toml.load('config_mil.toml')
    hp = custom_hp if custom_hp else config['hp']
    hp['pretraining'] = True if model_path else False
    print('--------Configuration---------- \n ', config)
    shutil.copy2('config_mil.toml', RES_PATH)

    device = torch.device(config['gpu_name'] if torch.cuda.is_available() else "cpu")
    print(f'Working on {base_path}!')
    print(f'using device {device}')

    aug_pipeline_train = get_training_pipeline(0, hp['crop_size'], mode='mil', strength=hp['aug_strength'],
                                               graham=hp['graham'])
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

    desc = f'_mil_pad_{str("_".join([k[0] + str(hp) for k, hp in hp.items()]))}_{os.path.basename(base_path)}'
    writer = Reporting(writer_desc=desc, log_dir=RES_PATH) if custom_writer is None else custom_writer
    best_model, perf_metric = train_model(net, criterion, optimizer_ft, plateau_scheduler, loaders, device,
                                          writer, num_epochs=num_epochs)
    return perf_metric


def setup_log(data_path):
    global RES_PATH
    RES_PATH = os.path.join(RES_PATH, f'{time.strftime("%Y%m%d_%H%M")}_{os.path.basename(data_path)}_PAD/')
    os.mkdir(RES_PATH)


def prepare_model(model_path, hp, device):
    net = None
    if hp['network'] == 'AlexNet':
        net = models.alexnet(pretrained=True)
        net.classifier[-1] = Linear(net.classifier[-1].in_features, 2)

    if hp['pretraining'] and hp['model_loading'] == 'features':
        net.load_state_dict(torch.load(model_path, map_location=device))
        print('Loaded stump: ', len(net.features))
    net = BagNet(net, num_attention_neurons=hp['attention_neurons'], attention_strategy=hp['attention'],
                 pooling_strategy=hp['pooling'], stump_type=hp['network'])
    if hp['pretraining'] and hp['model_loading'] == 'full':
        net.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        net.classifier = nn.Sequential(nn.Linear(net.L * net.K, 1), nn.Sigmoid())
        net.attention = nn.Sequential(nn.Linear(net.L, net.D), nn.Tanh(), nn.Linear(net.D, net.K))

    if hp['pretraining'] and hp['model_loading'] == 'extract':
        net.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        net = BagNet(net.stump, num_attention_neurons=hp['attention_neurons'], attention_strategy=hp['attention'],
                     pooling_strategy=hp['pooling'], stump_type=hp['network'])

    print(f'Model info: {net.__class__.__name__}, #frozen layer: {hp["freeze"]}')
    return net.to(device)


def train_model(model, criterion, optimizer, scheduler, loaders, device, writer, num_epochs=50):
    since = time.time()
    best_f1_val = -1
    model.to(device)
    val_scores, val_eye_scores = {}, {}
    perf_metrics = None

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
        writer.write_scores('train', train_scores, epoch)
        val_loss, perf_metrics = validate(model, criterion, loaders[1], device, writer, epoch)
        val_scores = perf_metrics.calc_scores(as_dict=True)

        best_f1_val = val_scores['f1'] if val_scores['f1'] > best_f1_val else best_f1_val

        scheduler.step(val_loss)

    time_elapsed = time.time() - since
    print(f'{time.strftime("%H:%M:%S")}> Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    torch.save(model.state_dict(), join(RES_PATH, f'model_pad_mil.pth'))
    return model, perf_metrics


def validate(model, criterion, loader, device, writer, cur_epoch):
    model.eval()
    running_loss = 0.0
    perf_metrics = Scores()

    for i, batch in tqdm(enumerate(loader), total=len(loader), desc='Validation'):
        inputs = batch['frames'].to(device, dtype=torch.float)
        labels = batch['label'].to(device)
        names = batch['name']   # pid for eye_id without L/R
        # positions = batch['pos']

        with torch.no_grad():
            loss, attention_weights, probs = model.calculate_objective(inputs, labels)
            error, preds = model.calculate_classification_error(inputs, labels)
            running_loss += loss.item()

        perf_metrics.add(preds, labels, probs=probs, tags=names, attention=attention_weights)

    scores = perf_metrics.calc_scores(as_dict=True)
    scores['loss'] = running_loss / len(loader.dataset)
    scores_eye = perf_metrics.calc_scores_eye(as_dict=True, ratio=0.5)
    writer.write_scores('eye_val', scores_eye, cur_epoch)
    writer.write_scores('val', scores, cur_epoch)
    perf_metrics.persist_scores(RES_PATH, cur_epoch, scores)

    return running_loss / len(loader.dataset), perf_metrics


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
