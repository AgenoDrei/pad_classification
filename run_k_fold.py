from transfer_learning import run
import argparse
import sys
from os.path import join
import math

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PAD k-fold')
    parser.add_argument('--data', help='Path for training data', type=str)
    parser.add_argument('--gpu', help='GPU name', type=str, default='cuda:0')
    parser.add_argument('--bs', help='Batch size for training', type=int, default=8)
    parser.add_argument('--epochs', help='Number of training epochs', type=int, default=50)
    parser.add_argument('--model', help='Path to pretrained model', type=str)
    parser.add_argument('--folds', '-k', help='number of folds', type=int)

    args = parser.parse_args()
    
    avg_f1 = 0
    f1_list = []
    for i in range(args.folds):
        f1 = run(join(args.data, f'fold{i}'), args.model, args.gpu, args.bs, args.epochs, 28)
        f1_list.append(f1)
        avg_f1 += f1 / args.folds
    
    std = math.sqrt(sum([(f - avg_f1)**2 / (len(f1_list)-1) for f in f1_list]))
    print('Avg f1 score for the PAD dataset: ', avg_f1)
    print('Standard divation for PAD dataset: ', std)
    sys.exit(0)



