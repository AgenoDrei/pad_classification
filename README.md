# Source code for the paper "Automated Detection of Peripheral Artery Disease From High Resolution Color Fundus Photographs" 
by Mueller, S et al.

## File structure
* include/ - scirpts providing classes for the creation of models, datasets and general convenience functions
* scripts/ - standalone scripts that are used to prepare the data or generate visualizations
* run_k_fold.py - main script to run the training using cross-validation
* search_hyperparameters.py - script to search for optimal hyperparameters across one fold in the dataset
* multiple_instance_learning.py - script to run the training process for the MIL model on a training and validation set
* prepare_data.py - script to convert the raw dataset into folds and generate label files for the training

All training scripts are configured using the *.toml files. 

## Example
Example for executing the k-fold training
```
python run_k_fold.py --data <path_to_data> --epochs 20 -k 7 -s "MIL" --model <path_to_pretrained_model>
```
