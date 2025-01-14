import json
import torch
import argparse
import data_utils
import model_utils
from PIL import Image

def predict(image_path, model, cat_to_name, topk=5):
    """Predict the class (or classes) of an image using a trained deep learning model.
    
    Args:
    image_path (str): Path to the image file.
    model (torch.nn.Module): Trained model to be used for prediction.
    cat_to_name (dict): Dictionary mapping class indices to flower names.
    topk (int): Number of top most likely classes to return.
    
    Returns:
    list of tuples: Each tuple contains the flower name and probability for the top predictions.
    """
    try:
        np_image = data_utils.process_image(image_path)
        if np_image is None:
            raise ValueError("Image processing failed.")
        
        image_tensor = torch.from_numpy(np_image).type(torch.FloatTensor)
        image_tensor = image_tensor.unsqueeze(0)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            output = model(image_tensor)
        
        probabilities = torch.exp(output)
        top_probs, top_classes = probabilities.topk(topk, dim=1)
        
        top_probs = top_probs.cpu().numpy().tolist()[0]
        top_classes = top_classes.cpu().numpy().tolist()[0]
        
        if model.class_to_idx is None:
            raise ValueError("model.class_to_idx is None")
        
        class_to_idx_inverted = {v: k for k, v in model.class_to_idx.items()}
        top_labels = [class_to_idx_inverted[i] for i in top_classes]
        top_flower_names = [cat_to_name.get(label, "Unknown") for label in top_labels]
        
        results = list(zip(top_flower_names, top_probs))
        return results

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return []

def check_device(gpu_flag):
    """Check for available GPUs/CPUs and print the information to the screen.
    
    Args:
    gpu_flag (bool): Flag indicating whether to use GPU if available.
    
    Returns:
    torch.device: The device to be used for prediction.
    """
    try:
        if torch.cuda.is_available() and gpu_flag:
            print('-' * 65)
            print(f"{torch.cuda.device_count()} GPU(s) available, shown below:\n")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            device = torch.device("cuda:0")
            print(f"\nUsing {torch.cuda.get_device_name(i)} for the prediction...")
            print('-' * 65)
        else:
            device = torch.device("cpu")
            print("GPU is not available or not selected. Using CPU for the code run.")
        return device
    except Exception as e:
        print(f"An error occurred while checking devices: {e}")
        return torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image along with the probability of that name')
    parser.add_argument('input', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, help='Return top K most likely classes', default=5)
    parser.add_argument('--category_names', type=str, help='Path to the JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')

    args = parser.parse_args()

    def display_prediction_banner():
        banner = """
    ################################################
    #                                              #
    #             Image Classification             #
    #               Model Prediction               #
    #                                              #
    ################################################
    
    Welcome to the Image Classification Model Prediction Solution, created by Seyi Ayobami Ayeni! 🌼

    This solution leverages the trained deep learning model to predict the class of a given image.
    You can use various pre-trained models, including VGG13, VGG16, and DenseNet121.
    Simply provide the path to the image and other parameters, and the model will predict the top 
    probable classes along with their probabilities.

    """

        print(banner)

    # Call the function before displaying commencing prediction
    display_prediction_banner()
    

    try:
        # Load the checkpoint
        model, optimizer, epochs = model_utils.load_checkpoint(args.checkpoint)
        if model is None:
            raise ValueError("Failed to load model checkpoint.")
        
        # Load the category names if provided
        cat_to_name = {}
        if args.category_names:
            with open(args.category_names, 'r') as f:
                cat_to_name = json.load(f)
        
        # Check and set the device
        device = check_device(args.gpu)
        model = model.to(device)

        # Predict the image
        results = predict(args.input, model, cat_to_name, topk=args.top_k)
        if not results:
            raise ValueError("Prediction failed.")
        
        # Print the most likely class and its probability
        print('-' * 65)
        most_likely_class, highest_probability = results[0]
        print(f"The most likely class is {most_likely_class} and its probability is {highest_probability:.4f}.")

        # Print all classes and their probability values
        print('-' * 65)
        print("Please find below the predicted labels and probabilities:\n")
        for name, prob in results:
            print(f"{name}: {prob:.4f}")
        print('-' * 65)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()