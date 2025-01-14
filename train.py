import os
import torch
import argparse
import data_utils
import model_utils
from torch import nn, optim
from datetime import datetime
from model_utils import build_model

def train(model, dataloaders, criterion, optimizer, dataset_sizes, device, epochs=5):
    """Train the model on the training data and validate it on the validation data.
    
    Args:
    model (torch.nn.Module): The neural network model to be trained.
    dataloaders (dict): Dictionary containing 'train', 'val', and 'test' dataloaders.
    criterion (torch.nn.Module): The loss function to optimize.
    optimizer (torch.optim.Optimizer): The optimizer to use for weight updates.
    dataset_sizes (dict): Dictionary containing sizes of 'train', 'val', and 'test' datasets.
    device (torch.device): Device on which the model is being run.
    epochs (int): The number of epochs to train the model.
    
    Returns:
    tuple: Lists of training and validation losses, and accuracies.
    """
    try:
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 60)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                if phase == 'train':
                    train_losses.append(epoch_loss)
                    train_accuracies.append(epoch_acc)
                else:
                    val_losses.append(epoch_loss)
                    val_accuracies.append(epoch_acc)

                print(f'{phase} Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}')

        return train_losses, val_losses, train_accuracies, val_accuracies
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return None, None, None, None

def check_device(gpu_flag):
    """Check for available GPUs/CPUs and print the information to the screen.
    
    Args:
    gpu_flag (bool): Flag indicating whether to use GPU if available.
    
    Returns:
    torch.device: The device to be used for training and prediction.
    """
    try:
        if torch.cuda.is_available() and gpu_flag:
            print('-' * 60)
            print(f"{torch.cuda.device_count()} GPU(s) available, shown below:\n")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            device = torch.device("cuda:0")
            print(f"\nUsing {torch.cuda.get_device_name(i)} for the training...")
            print('-' * 60)
        else:
            device = torch.device("cpu")
            print("GPU is not available or not selected. Using CPU for the code run.")
        return device
    except Exception as e:
        print(f"An error occurred while checking devices: {e}")
        return torch.device("cpu")

def calculate_average_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    """Calculate and print average training and validation losses and accuracies.
    
    Args:
    train_losses (list): List of training losses for each epoch.
    val_losses (list): List of validation losses for each epoch.
    train_accuracies (list): List of training accuracies for each epoch.
    val_accuracies (list): List of validation accuracies for each epoch.
    
    Returns:
    None
    """
    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_val_loss = sum(val_losses) / len(val_losses)
    avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)
    avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)
    
    print(f"Average Training Loss: {avg_train_loss:.4f}")
    print(f"Average Validation Loss: {avg_val_loss:.4f}")
    print(f"Average Training Accuracy: {avg_train_accuracy:.4f}")
    print(f"Average Validation Accuracy: {avg_val_accuracy:.4f}")
    return avg_train_loss, avg_val_loss, avg_train_accuracy, avg_val_accuracy

def main():
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint')
    parser.add_argument('data_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, help='Full path to save checkpoints to', default='model_checkpoints')
    parser.add_argument('--arch', type=str, help='Model architecture', default='vgg16')
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.001)
    parser.add_argument('--hidden_units', type=int, help='Number of hidden units', default=4096)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=10)
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available (default: False)')

    args = parser.parse_args()

    def display_training_banner():
        banner = """
    ##############################################
    #                                            #
    #            Image Classification            #
    #                 Model Training             #
    #                                            #
    ##############################################
    
    Welcome to the Image Classification Model Training Solution, created by Seyi Ayobami Ayeni! 🎉

    This solution leverages state-of-the-art deep learning architectures to classify images into
    various categories.
    You can choose from multiple pre-trained models, including VGG13, VGG16, and DenseNet121.
    The solution is flexible and efficient, providing you with the tools to train, evaluate, and save your
    models seamlessly.

    """

        print(banner)

    # Call the function before displaying the training arguments
    display_training_banner()

    # Ensure the save directory exists before training starts
    try:
        os.makedirs(args.save_dir, exist_ok=True)
    except Exception as e:
        print(f"Error: Could not create or access save directory {args.save_dir}.")
        print(e)
        return

    # Build the model
    model = build_model(arch=args.arch, hidden_units=args.hidden_units)

    # Ensure the arch attribute is set on the model
    model.arch = args.arch

    print("\nThe following arguments have been entered for the training:")
    print('-' * 60)
    print(f"data_dir: {args.data_dir}")
    print(f"save_dir: {args.save_dir}")
    print(f"arch: {args.arch}")
    print(f"input_units: {args.input_units}")
    print(f"learning_rate: {args.learning_rate}")
    print(f"hidden_units: {args.hidden_units}")
    print(f"epochs: {args.epochs}")
    print(f"gpu: {args.gpu}")
    print('-' * 60)


    try:
        # Initialize data
        data_transforms, image_datasets, dataloaders, dataset_sizes, class_names = data_utils.initialize_data(args.data_dir)
        if data_transforms is None:
            raise ValueError("Data initialization failed.")

        # Build the model
        model = model_utils.build_model(arch=args.arch, hidden_units=args.hidden_units)
        model.class_to_idx = image_datasets['train'].class_to_idx

        # Check and set the device
        device = check_device(args.gpu)
        model = model.to(device)

        # Define criterion and optimizer
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

        # Train the model
        train_losses, val_losses, train_accuracies, val_accuracies = train(model, dataloaders, criterion, optimizer, dataset_sizes, device, epochs=args.epochs)
        if train_losses is None:
            raise ValueError("Model training failed.")

        # Calculate and print average metrics
        calculate_average_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

        avg_train_loss, avg_val_loss, avg_train_accuracy, avg_val_accuracy = calculate_average_metrics(train_losses, val_losses, train_accuracies, val_accuracies)


        # Generate the filename with the model name, timestamp, and accuracy
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = model.arch.replace('_', '-')
        accuracy_str = f'val_acc_{int(avg_val_accuracy * 100)}%'
        checkpoint_filename = f'{model_name}_{timestamp}_{accuracy_str}.pth'
        
        # Specify the path
        checkpoint_path = os.path.join(args.save_dir, checkpoint_filename)

        model_utils.save_checkpoint(model, optimizer, args.epochs, path=checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()