import os
import torch
import torch.nn as nn
from torchvision import models
from torch.serialization import add_safe_globals
from torch.nn import Sequential, Linear, ReLU, Dropout, LogSoftmax

def build_model(arch='vgg16', input_units=None, hidden_units=4096, output_units=102, dropout=0.5):
    """Build a pre-trained model with a custom classifier.
    
    Args:
    arch (str): Architecture of the pre-trained model.
    input_units (int, optional): Number of input units. If None, inferred from the pre-trained model.
    hidden_units (int): Number of units in the hidden layer.
    output_units (int): Number of output units (number of classes).
    dropout (float): Dropout rate for the dropout layer.
    
    Returns:
    torch.nn.Module: The built model.
    """
    if arch == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        expected_input_units = model.classifier[0].in_features
    elif arch == 'vgg13':
        model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
        expected_input_units = model.classifier[0].in_features    
    elif arch == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        expected_input_units = model.classifier.in_features
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    if input_units is not None and input_units != expected_input_units:
        raise ValueError(f"Specified input_units {input_units} does not match the expected {expected_input_units} for architecture {arch}")
    
    input_units = expected_input_units if input_units is None else input_units
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Custom classifier
    classifier = nn.Sequential(
        nn.Linear(input_units, hidden_units),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_units, output_units),
        nn.LogSoftmax(dim=1)
    )

    if arch in 'densenet121':
        model.classifier = classifier
    else:
        model.classifier = classifier

    model.arch = arch
    
    return model

def save_checkpoint(model, optimizer, epochs, path='model_checkpoints/checkpoint0001.pth'):
    """Save the trained model and necessary information for future inference or continued training.
    
    Args:
    model (torch.nn.Module): The trained model.
    optimizer (torch.optim.Optimizer): The optimizer used for training.
    epochs (int): The number of epochs.
    path (str): The path to save the checkpoint file.
    
    Returns:
    None
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs,
        'arch': model.arch,
        'classifier': model.classifier if hasattr(model, 'classifier') else model.fc,
        'class_to_idx': model.class_to_idx  # Ensure class_to_idx is saved
    }
    
    torch.save(checkpoint, path)

# Allowlist Sequential, Linear, and set for safe loading
add_safe_globals([Sequential, Linear, ReLU, Dropout, LogSoftmax, set])
    
def load_checkpoint(filepath):
    """Load a checkpoint and rebuild the model.
    
    Args:
    filepath (str): Path to the checkpoint file.
    
    Returns:
    tuple: The rebuilt model, optimizer, and number of epochs.
    """
    # Allowlist necessary classes/functions
    add_safe_globals([Sequential, Linear, ReLU, Dropout, LogSoftmax, set])
    
    checkpoint = torch.load(filepath, weights_only=True)
    
    model = build_model(arch=checkpoint['arch'], input_units=checkpoint.get('input_units', None), hidden_units=4096, output_units=102)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint.get('class_to_idx', None)  # Ensure class_to_idx is loaded
    
    optimizer = torch.optim.Adam(model.classifier.parameters() if hasattr(model, 'classifier') else model.fc.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epochs = checkpoint['epochs']

    # Print the loaded checkpoint message
    print(f"Loaded checkpoint '{filepath}' with {epochs} epochs.")
    
    return model, optimizer, epochs