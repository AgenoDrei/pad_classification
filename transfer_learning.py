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

from include.nn_datasets import RetinaDataset, SegmentsDataset, get_validation_pipeline, get_training_pipeline
from include.nn_utils import dfs_freeze, Scores, write_scores


RES_PATH = ''


def run(base_path, model_path, num_epochs):
    setup_log(base_path)
    # load hyperparameter
    config = toml.load('config.toml')
    hp = config['hp']
    hp['pretraining'] = True if model_path else False
    print('--------Configuration---------- \n ', config)

    device = torch.device(config['gpu_name'] if torch.cuda.is_available() else "cpu")
    print(f'Working on {base_path}!')
    print(f'using device {device}')

    aug_pipeline_train = get_training_pipeline(hp['image_size'], hp['crop_size'])
    aug_pipeline_val = get_validation_pipeline(hp['image_size'], hp['crop_size'])

    loaders = prepare_dataset(base_path, hp, aug_pipeline_train, aug_pipeline_val, config['num_workers'],
                              config['eye_segments'])
    net = prepare_model(model_path, hp, device)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=hp['learning_rate'],
                              weight_decay=hp['weight_decay'])
    criterion = CrossEntropyLoss()
    plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=5, verbose=True)

    desc = f'_transfer_pad_{str("_".join([k[0] + str(hp) for k, hp in hp.items()]))}'
    writer = SummaryWriter(comment=desc)
    best_model, scores, eye_scores = train_model(net, criterion, optimizer_ft, plateau_scheduler, loaders, device, writer,
                                     num_epochs=num_epochs, description=desc)
    return scores, eye_scores


def setup_log(data_path):
    global RES_PATH
    RES_PATH = f'{time.strftime("%Y%m%d_%H%M")}_{os.path.basename(data_path)}_PAD/'
    os.mkdir(RES_PATH)


def prepare_model(model_path, hp, device):
    net = None
    if hp['network'] == 'AlexNet':
        net = models.alexnet(pretrained=True)
        num_ftrs = net.classifier[-1].in_features
        net.classifier[-1] = Linear(num_ftrs, 2)
    elif hp['network'] == 'Inception':
        net = inceptionv4()
        num_ftrs = net.last_linear.in_features
        net.last_linear = Linear(num_ftrs, 2)
        for i, child in enumerate(net.features.children()):
            if i < len(net.features) * hp['freeze']:
                for param in child.parameters():
                    param.requires_grad = False
                dfs_freeze(child)

    if hp['pretraining']:
        net.load_state_dict(torch.load(model_path, map_location=device))
        # net.last_linear = Linear(net.last_linear.in_features, 5)
    # net.train()

    print(
        f'Model info: {net.__class__.__name__}, layer: {len(net.features)}, #frozen layer: {len(net.features) * hp["freeze"]}')
    return net


def prepare_dataset(base_name: str, hp, aug_pipeline_train, aug_pipeline_val, num_workers, eye_segments):
    set_names = ('train', 'val')
    dataset = RetinaDataset if eye_segments == 1 else SegmentsDataset
    train_dataset = dataset(join(base_name, 'labels_train.csv'), join(base_name, set_names[0]),
                            augmentations=aug_pipeline_train, file_type='.jpg', use_prefix=False, class_iloc=1,
                            thresh=hp['class_threshold'])
    val_dataset = dataset(join(base_name, 'labels_val.csv'), join(base_name, set_names[1]),
                          augmentations=aug_pipeline_val, file_type='.jpg', use_prefix=False, class_iloc=1,
                          thresh=hp['class_threshold'])

    sample_weights = [train_dataset.get_weight(i) for i in range(len(train_dataset))]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=False,
                                               sampler=sampler, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hp['batch_size'], shuffle=False,
                                             num_workers=num_workers)
    print(f'Dataset ({dataset.__name__}) info:\n Train size: {len(train_dataset)},\n Validation size: {len(val_dataset)}')
    return train_loader, val_loader


def train_model(model, criterion, optimizer, scheduler, loaders, device, writer, num_epochs=50, description='Vanilla'):
    since = time.time()
    best_f1_val = -1
    model.to(device)

    for epoch in range(num_epochs):
        print(f'{time.strftime("%H:%M:%S")}> Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        running_loss = 0.0
        metrics = Scores()
        # Iterate over data.
        for i, batch in tqdm(enumerate(loaders[0]), total=len(loaders[0]), desc=f'Epoch {epoch}'):
            inputs = batch['image'].to(device, dtype=torch.float)
            labels = batch['label'].to(device)

            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            metrics.add(pred, labels)

        train_scores = metrics.calc_scores(as_dict=True)
        train_scores['loss'] = running_loss / len(loaders[0].dataset)
        write_scores(writer, 'train', train_scores, epoch)
        val_loss, val_scores, val_eye_scores = validate(model, criterion, loaders[1], device, writer, epoch)

        best_f1_val = val_scores['f1'] if val_scores['f1'] > best_f1_val else best_f1_val

        scheduler.step(val_loss)

    time_elapsed = time.time() - since
    print(
        f'{time.strftime("%H:%M:%S")}> Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s with best f1 score of {best_f1_val}')

    # validate(model, criterion, loaders[1], device, writer, num_epochs, calc_roc=True)
    torch.save(model.state_dict(), join(RES_PATH, f'model_pad_transfer.pth'))

    return model, val_scores, val_eye_scores


def validate(model, criterion, loader, device, writer, cur_epoch, calc_roc=False) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    perf_metrics = Scores()
    sm = torch.nn.Softmax(dim=1)

    for i, batch in tqdm(enumerate(loader), total=len(loader), desc='Validation'):
        inputs = batch['image'].to(device, dtype=torch.float)
        labels = batch['label'].to(device)
        names = batch['eye']

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            probs = sm(outputs)
            running_loss += loss.item() * inputs.size(0)

        perf_metrics.add(preds, labels, probs=probs, tags=names)

    scores = perf_metrics.calc_scores(as_dict=True)
    scores_eye = perf_metrics.calc_scores_eye(as_dict=True, ratio=0.5)
    scores['loss'] = running_loss / len(loader.dataset)
    write_scores(writer, 'val', scores, cur_epoch, full_report=True)
    write_scores(writer, 'eye_val', scores_eye, cur_epoch, full_report=True)
    perf_metrics.data.to_csv(join(RES_PATH, f'{cur_epoch}_last_pad_model_{scores["f1"]:0.3}.csv'), index=False)

    return running_loss / len(loader.dataset), scores, scores_eye


if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using torch with GPU {torch.cuda.is_available()}')

    parser = argparse.ArgumentParser(description='Pretraing but with crops (glutenfree)')
    parser.add_argument('--data', help='Path for training data', type=str)
    parser.add_argument('--epochs', help='Number of training epochs', type=int, default=50)
    parser.add_argument('--model', help='Path to pretrained model', type=str, default=None)
    args = parser.parse_args()

    run(args.data, args.model, args.epochs)
    sys.exit(0)
