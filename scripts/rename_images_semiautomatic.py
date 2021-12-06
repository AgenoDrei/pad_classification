import os
import click
import shutil
import pandas as pd
from fuzzywuzzy import process, fuzz
from os.path import join


@click.command()
@click.option('--input_path', '-i', type=str, required=True)
@click.option('--output_path', '-o', type=str, default='results/')
@click.option('--labels_path', '-l', type=str, required=True)
def run(input_path, output_path, labels_path):
    shutil.rmtree(output_path, ignore_errors=True)
    os.mkdir(output_path)
    df = pd.read_excel(labels_path, sheet_name='f_R', header=[0, 1, 2, 3])
    images = os.listdir(input_path)
    print(f'Found {len(images)} files')

    # Create lookup table for name - id
    names = {'L': {}, 'R': {}}
    for row in df.itertuples(index=False):
        name = f'{row[3]}  {row[2]}'
        if row[6] not in 'R' and row[6] not in 'L':
            continue
        names[row[6]][name] = row[0]

    # match file names with label names
    moved_images, duplicates = [], []
    for file in images:
        print(f'Working on {file}...')
        filename = os.path.splitext(os.path.basename(file))[0]
        parts = filename.split('--')
        if len(parts) < 2:
            print(f'Error in splitting for {filename}!')
            return
        name_part, suffix_part = parts
        if 'OD' not in suffix_part and 'OS' not in suffix_part:
            print(f'Error in filename for {filename}!')
            return
        # Extract eye (left / right) and patient name from the file
        eye = 'R' if 'OD' in suffix_part else 'L'
        name_part = name_part.replace('-', ' ').replace('__', '??').replace('_', ' ')
        print(f'Extracted name: {name_part}')

        # Find match between filename and patient list
        best_match = process.extractOne(name_part, [k for k, v in names[eye].items()], scorer=fuzz.token_set_ratio)
        best_match_id = names[eye][best_match[0]]
        print(f'Best match from labels file: {best_match} with id: {best_match_id}')
        if best_match[1] < 70:
            print(f'No sufficient match found for {filename}!')
            return
        # Persist renaming on disk (patient name -> patient id)
        shutil.copy(join(input_path, file), join(output_path, f'{best_match_id}{os.path.splitext(file)[1]}'))
        print(f'Moved {file} to {output_path}{best_match_id}{os.path.splitext(file)[1]}.')
        if best_match_id in moved_images:
            duplicates.append(best_match[0])
            print(f'Found duplicate for id: {best_match_id}, file: {file}')
        moved_images.append(best_match_id)
        print()
    print(f'Processing done. Found duplicates: {duplicates}')


if __name__ == '__main__':
    run()


