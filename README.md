# Flower Classifier Project

## Overview
This solution trains a convolutional neural network (CNN) model to classify images of flowers into their respective categories. The solution includes training a model on a dataset of flower images, saving the trained model as a checkpoint, and making predictions using the trained model. The project uses Python, PyTorch, and pre-trained neural networks from the torchvision library.

## Project Structure

<pre style="text-align: left;">
project_root/
├── data_utils.py    # Utility functions for loading data and preprocessing images
├── model_utils.py   # Functions and classes related to the model
├── train.py         # Script for training the model
├── predict.py       # Script for making predictions with the trained model
├── cat_to_name.json # JSON file mapping categories to real names
├── save_dir         # directory to load files from with sub-directories for train, test, and validation 
└── README.md        # Project documentation and instructions
</pre>

## Requirements
- Python 3.x
- PyTorch
- torchvision
- PIL
- numpy
- argparse
- json

You can install the required packages using:
```bash
pip install torch torchvision pillow numpy argparse
```

## Training the Model
The train.py script is used to train a new neural network on a dataset and save the trained model as a checkpoint.

### Usage

```bash
python train.py [-h] [--save_dir SAVE_DIR] [--arch ARCH] [--hidden_units HIDDEN_UNITS] [--learning_rate LEARNING_RATE] [--epochs EPOCHS] [--gpu] data_dir
```

## Arguments

data_dir: Directory of the dataset (There is not default. User must specify it at runtime and the folder must contain directories named 'train', 'valid', and 'test' containing train, validation, and test datasets respectively).

- --save_dir SAVE_DIR: Directory to save checkpoints (Directory path to save checkpoints to).

- --arch ARCH: Model architecture (default: vgg16).

- --hidden_units HIDDEN_UNITS: Number of hidden units (default: 4096).

- --learning_rate LEARNING_RATE: Learning rate (default: 0.001).

- --epochs EPOCHS: Number of epochs (default: 10).

- --gpu: Use GPU if available (default: False).


## Example Command

```bash
python train.py --save_dir model_checkpoints --arch vgg16 --hidden_units 4096 --learning_rate 0.001 --epochs 10 --gpu flowers
```

## Predicting with the Trained Model

The predict.py script is used to predict the class of a flower image using a trained model checkpoint.

### Usage

```bash
python predict.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES] [--gpu] input checkpoint
```

## Arguments

- input: Path to the input image (required).

- checkpoint: Path to the model checkpoint file (required).

- --top_k TOP_K: Number of top most likely classes to return (default: 5).

- --category_names CATEGORY_NAMES: Path to the JSON file mapping categories to real names.

- --gpu: Use GPU for inference if available.


## Example Command

```bash
python predict.py --top_k 5 --category_names cat_to_name.json --gpu assets/img-check.jpg model_checkpoints/vgg16_20250114_071550_val_acc_72%.pth
```

## Functionality

### data_utils.py

- initialize_data(data_dir): Initialize data transforms, datasets, dataloaders, and related variables.

- process_image(image_path): Scales, crops, and normalizes a PIL image for a PyTorch model, returns a Numpy array.

### model_utils.py

- build_model(arch='vgg16', input_units=None, hidden_units=4096, output_units=102, dropout=0.5): Build a pre-trained model with a custom classifier.

- save_checkpoint(model, optimizer, epochs, path='checkpoint.pth'): Save the trained model and necessary information for future inference or continued training.

- load_checkpoint(filepath): Load a checkpoint and rebuild the model.


### train.py

- train(model, dataloaders, criterion, optimizer, dataset_sizes, device, epochs=10): Train the model on the training data and validate it on the validation data.

- check_device(gpu_flag): Check for available GPUs/CPUs and print the information to the screen.

- calculate_average_metrics(train_losses, val_losses, train_accuracies, val_accuracies): Calculate and print average training and validation losses and accuracies.

### predict.py
- predict(image_path, model, cat_to_name, topk=5): Predict the class (or classes) of an image using a trained deep learning model.

- check_device(gpu_flag): Check for available GPUs/CPUs and print the information to the screen.


## License
This project is licensed under the MIT License. See the LICENSE file for more details.


- The dataset distribution:
![dataset_distribution](https://github.com/user-attachments/assets/896af2c7-0d4e-442c-b895-07a8ba0e4868)

- Training process:
![image](https://github.com/user-attachments/assets/53c7d37e-ae95-4153-b345-42831b184a12)

-  Sample prediction:
![predictions](https://github.com/user-attachments/assets/a11fc24c-a5cd-4cb3-8c23-0fa52206503e)

- Command-line operations:
![pg1](https://github.com/user-attachments/assets/c7d476b6-0dd9-49ab-8f88-67197addb6b6)
![pg2](https://github.com/user-attachments/assets/63044061-da78-4606-85ba-aa40e7f39d78)
![pg3](https://github.com/user-attachments/assets/565006c9-fa4c-41e7-a984-7f4e53fdcfd5)
![pg4](https://github.com/user-attachments/assets/50c1e640-8a9b-4ee6-bc06-139e0fbf5d31)
![pg5](https://github.com/user-attachments/assets/fb0014c1-327e-4ba3-87f3-49b671e50297)






