import pandas as pd
import shutil
import os
import click
from os.path import join


@click.command()
@click.option('--input_path', '-i', help='Input path')
@click.option('--output_path', '-o', help='Output path')
@click.option('--labels_path', '-l', help='Label file')
def run(input_path, output_path, labels_path):
    df = pd.read_csv(labels_path)
    df = df.fillna(0)
    
    [os.mkdir(join(output_path, str(i))) for i in range(1, 6)]

    for row in df.itertuples():
        if row[4] == 0:
            continue
        shutil.copy2(join(input_path, row[1] + '.jpg'), join(output_path, str(int(row[4])), row[1] + '.jpg'))


if __name__ == '__main__':
    run()
