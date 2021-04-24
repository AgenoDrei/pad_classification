import os
import cv2
import matplotlib.pyplot as plt
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
    return fig

# Helper function to show a batch
def show_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, label_batch = sample_batched['image'], sample_batched['label']
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
