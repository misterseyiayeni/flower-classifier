import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

def initialize_data(data_dir):
    """Initialize data transforms, datasets, dataloaders, and related variables.
    
    Args:
    data_dir (str): Path to the main data directory.
    
    Returns:
    tuple: Data transforms, image datasets, dataloaders, dataset sizes, and class names.
    """
    try:
        train_dir = os.path.join(data_dir, 'train')
        valid_dir = os.path.join(data_dir, 'valid')
        test_dir = os.path.join(data_dir, 'test')

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        image_datasets = {
            'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
            'val': datasets.ImageFolder(valid_dir, transform=data_transforms['val']),
            'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
        }

        dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
            'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=4),
            'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=4),
        }

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
        class_names = image_datasets['train'].classes

        return data_transforms, image_datasets, dataloaders, dataset_sizes, class_names

    except Exception as e:
        print(f"An error occurred while initializing data: {e}")
        return None, None, None, None, None

def process_image(image_path):
    """Scales, crops, and normalizes a PIL image for a PyTorch model, returns a Numpy array.
    
    Args:
    image_path (str): Path to the image file.
    
    Returns:
    np.array: Processed image as a Numpy array.
    """
    try:
        image = Image.open(image_path)
        aspect = image.size[0] / float(image.size[1])
        if aspect > 1:
            image = image.resize((int(aspect * 256), 256))
        else:
            image = image.resize((256, int(256 / aspect)))
        
        width, height = image.size
        new_width, new_height = 224, 224
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        image = image.crop((left, top, right, bottom))
        
        np_image = np.array(image) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_image = (np_image - mean) / std
        np_image = np_image.transpose((2, 0, 1))
        
        return np_image

    except Exception as e:
        print(f"An error occurred while processing the image: {e}")
        return None