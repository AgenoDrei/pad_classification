import argparse
import torch
import sys
from torch import nn
from include.vis.gradcam import GradCam
from include.vis.misc_functions import save_class_activation_images
from pretrainedmodels import inceptionv4

# ToDo: Load images from only the positve class
def run(data_path, model_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = inceptionv4()
    model.last_linear = nn.Linear(model.last_linear.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))

    grad_cam = GradCam(model, target_layer=22)
    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img, target_class)
    # Save mask
    save_class_activation_images(original_image, cam, file_name_to_export)
    print('Grad cam completed')
    print(f'Using device {device}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize actiavtions in known images')
    parser.add_argument('--data', help='Path for image data', type=str)
    parser.add_argument('--model', help='Path to pretrained model', type=str)
    args = parser.parse_args()

    run(args.data, args.model)
    sys.exit(0)
