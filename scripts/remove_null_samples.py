import os
import click
import shutil
import pandas as pd
from os.path import join


@click.command()
@click.option('--input_path', '-i', type=str, required=True)
@click.option('--labels_path', '-l', type=str, required=True)
def run(input_path, labels_path):
    df = pd.read_csv(labels_path)
    clean_df = pd.DataFrame(columns=df.columns)
    images = os.listdir(input_path)
    images = [os.path.splitext(img)[0] for img in images]
    print(f'Found {len(images)} files')

    for row in df.itertuples(index=False):
        if row[0] not in images:
            print(f'{row[0]} not found.')
            continue
        clean_df = clean_df.append(row._asdict(), ignore_index=True)

    clean_df.to_csv(f'clean_{labels_path}', index=False)
    print(f'Created labels.csv with {len(clean_df)} entries.')

if __name__ == '__main__':
    run()


