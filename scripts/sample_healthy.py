import os
import click
import shutil
import pandas as pd
from os.path import join


@click.command()
@click.option('--input_path', '-i', type=str, required=True)
@click.option('--labels_path', '-l', type=str, required=True)
@click.option('--output_path', '-o', type=str, default='results/')
@click.option('--num_samples', '-n', type=int, default=70)
def run(input_path, labels_path, output_path, num_samples):
    shutil.rmtree(output_path, ignore_errors=True)
    os.mkdir(output_path)
    df = pd.read_csv(labels_path)
    df = df.sample(frac=1).reset_index(drop=True)
    clean_df = pd.DataFrame(columns=['ID', 'Eye', 'Group', 'pavk_FS_max'])

    images = os.listdir(input_path)
    print(f'Found {len(images)} files')

    count = 0
    prefix = 'XX'
    for row in df.itertuples(index=False):
        if count >= num_samples:
            break
        if row[1] == 1:
            continue
        sample_name, eye = row[0], 'X'
        new_name = f'{prefix}{count:04d}{eye}'
        new_entry = {'ID': new_name, 'Eye': eye, 'Group': 0, 'pavk_FS_max': 0}
        clean_df = clean_df.append(new_entry, ignore_index=True)
        shutil.copy2(join(input_path, sample_name + '.jpg'), join(output_path, new_name + '.jpg'))
        count += 1

    clean_df.to_csv(f'sampled_{labels_path}', index=False)
    print(f'Created sampled_labels.csv with {len(clean_df)} entries.')


if __name__ == '__main__':
    run()
