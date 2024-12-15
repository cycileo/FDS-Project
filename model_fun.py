import torch.optim as optim
import torch.nn as nn
import torch
import time
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import Counter
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, image_width, num_layers=2, first_layer_filters=16):
        super(SimpleCNN, self).__init__()

        self.num_layers = num_layers
        self.first_layer_filters = first_layer_filters

        layers = []
        in_channels = 3
        out_channels = first_layer_filters

        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Halves spatial dimensions

            in_channels = out_channels
            out_channels *= 2  # Double the number of filters for each subsequent layer

        layers.append(nn.Dropout(0.2))

        self.features = nn.Sequential(*layers)

        # Calculate the output size after the convolutional layers
        final_width = image_width // (2 ** num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * (final_width ** 2), 512),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

def train_model(model, num_classes, train_loader, val_loader, epoch=50):
    """
    Trains the given model using the provided training and validation datasets.

    The function performs the training loop, computes the loss using CrossEntropyLoss, and applies early stopping 
    if the validation loss does not improve for a specified number of epochs (patience).

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        num_classes (int): The number of output classes.
        train_loader (DataLoader): The DataLoader for the training dataset.
        val_loader (DataLoader): The DataLoader for the validation dataset.
        epoch (int): The maximum number of epochs to train the model. Default is 50.

    Returns:
        tuple: A tuple containing:
            - val_accuracies (list): Validation accuracy history.
            - val_losses (list): Validation loss history.
            - elapsed_time (float): Total training time in seconds.
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Set maximum number of epochs and patience for early stopping
    max_epochs = epoch
    patience = 5  # Number of epochs with no improvement after which training will stop
    best_val_loss = float('inf')  # Initialize best validation loss as infinity
    epochs_without_improvement = 0

    # Lists to store validation accuracy and loss for plotting later
    val_accuracies = []
    val_losses = []

    # Start timing the training process
    start_time = time.time()

    # Training loop
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        train_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", unit="batch")  # Add tqdm to training loop
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to device
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loader.set_postfix(loss=loss.item())  # Update tqdm progress bar with current loss
        
        print(f"Epoch {epoch+1}, Average Loss: {(running_loss / len(train_loader)):.4f}")

        # Evaluate on the validation dataset
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient computation during evaluation
            for images, labels in val_loader:  # Assuming val_loader is the DataLoader for the test/validation set
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Append validation metrics to the lists
        val_accuracies.append(accuracy)
        val_losses.append(avg_val_loss)

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0  # Reset counter
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break  # Stop training early if no improvement

        # Check if we have reached the maximum number of epochs
        if epoch + 1 == max_epochs:
            print("Maximum number of epochs reached.")
            break

    # End timing the training process
    elapsed_time = time.time() - start_time

    # Return validation accuracy and loss history, along with elapsed time
    return val_accuracies, val_losses, elapsed_time, epoch

def test_model(model, test_loader):
    """
    Evaluates the trained model on the test dataset and computes the test accuracy and F1 score.

    The function computes the model's accuracy on the test set and the weighted F1 score based on the predicted and 
    true labels. It also provides a progress bar using `tqdm`.

    Args:
        model (torch.nn.Module): The trained neural network model to be evaluated.
        test_loader (DataLoader): The DataLoader for the test dataset.

    Returns:
        None: The function prints the test accuracy and F1 score.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available

    # Evaluate on the test dataset with tqdm
    model.eval()
    correct = 0
    total = 0
    all_labels = []  # To store true labels
    all_preds = []   # To store predicted labels
    test_loader = tqdm(test_loader, desc="Testing", unit="batch")  # Add tqdm for the test loop

    with torch.no_grad():  # Disable gradient computation during evaluation
        for images, labels in test_loader:  # Loop through test dataset
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())  # Append true labels
            all_preds.extend(predicted.cpu().numpy())  # Append predicted labels
            
            # Update tqdm bar with current accuracy
            test_loader.set_postfix(accuracy=(100 * correct / total))  

    # Compute test accuracy
    test_accuracy = 100 * correct / total

    # Compute F1 score for the whole dataset
    f1 = f1_score(all_labels, all_preds, average='weighted')  # You can also use 'macro' or 'micro' as needed

    print(f"\nTest Accuracy: {test_accuracy:.2f}%")
    print(f"Test F1 Score (Weighted): {f1:.4f}")

def evaluate_model(model, test_loader, class_names):
    """
    Evaluates the model on the test dataset and computes the classification metrics.

    The function calculates the classification report (including precision, recall, F1 score) and the confusion matrix.
    It uses `tqdm` to display progress during evaluation.

    Args:
        model (torch.nn.Module): The trained neural network model to be evaluated.
        val_loader (DataLoader): The DataLoader for the test dataset.
        class_names (list of str): A list of class names corresponding to the output classes.

    Returns:
        tuple: A tuple containing:
            - confusion_matrix (np.ndarray): The confusion matrix of predicted vs true labels.
            - metrics_report (str): The classification report including precision, recall, and F1 score.
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics_report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    return cm, metrics_report

def load_model(file_path, num_classes, image_width, num_layers, first_layer_filters):
    """
    Creates a SimpleCNN model using the provided hyperparameters, loads the state dictionary
    from the given file path, and moves the model to the specified device.

    Args:
        file_path (str): Path to the saved model file (state_dict).
        num_classes (int): Number of classes for the model.
        image_width (int): Width of the input images.
        num_layers (int): Number of layers in the CNN.
        first_layer_filters (int): Number of filters in the first convolutional layer.
        device (torch.device): The device to move the model to (e.g., 'cuda' or 'cpu').

    Returns:
        model (torch.nn.Module): The loaded SimpleCNN model.
    """
    # Define the model
    model = SimpleCNN(
        num_classes=num_classes,
        image_width=image_width,
        num_layers=num_layers,
        first_layer_filters=first_layer_filters
    )

    # Load the state dictionary
    model.load_state_dict(torch.load(file_path))

    # Move the model to the specified device
    model = model.to(device)

    return model

def load_data(image_width, main_folder='PlantVillage', batch_size=32, num_workers=4):
    """
    Load and preprocess datasets for training, validation, and testing.
    
    Parameters:
    - image_width (int): The width (and height) to resize images to.
    - main_folder (str): The main folder path that contains 'train', 'val', and 'test' subfolders.
    - batch_size (int): Batch size for the DataLoader.
    - num_workers (int): Number of workers for DataLoader.

    Returns:
    - train_loader (DataLoader): DataLoader for the training dataset.
    - val_loader (DataLoader): DataLoader for the validation dataset.
    - test_loader (DataLoader): DataLoader for the test dataset.
    """
    # Define transformations for your dataset
    transform = transforms.Compose([
        transforms.Resize((image_width, image_width)),  # Resize all images
        transforms.ToTensor(),          # Convert images to PyTorch tensors
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    # Build dataset paths dynamically
    train_path = os.path.join(main_folder, 'train')
    val_path = os.path.join(main_folder, 'val')
    test_path = os.path.join(main_folder, 'test')

    # Load train, val, and test datasets
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

    # Create DataLoaders for train, val, and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = train_dataset.classes

    return train_loader, val_loader, test_loader, class_names

def load_test_data(image_width, test_folder='PlantDiseases', batch_size=32, num_workers=4):
    """
    Load and preprocess the test dataset.

    Parameters:
    - image_width (int): The width (and height) to resize images to.
    - test_folder (str): The folder path containing the test dataset.
    - batch_size (int): Batch size for the DataLoader.
    - num_workers (int): Number of workers for DataLoader.

    Returns:
    - test_loader (DataLoader): DataLoader for the test dataset.
    - class_names (list): List of class names from the dataset.
    """

    # Define transformations for the test dataset
    transform = transforms.Compose([
        transforms.Resize((image_width, image_width)),  # Resize all images
        transforms.ToTensor(),          # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    # Load the test dataset
    test_dataset = datasets.ImageFolder(root=test_folder, transform=transform)

    # Create DataLoader for the test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Extract class names
    class_names = test_dataset.classes

    return test_loader, class_names

def plot_training_epochs_hystory(val_accuracies, val_losses, epochs):
    """
    Plots validation accuracies and losses over epochs.

    Args:
        val_accuracies (list): List of validation accuracies over epochs.
        val_losses (list): List of validation losses over epochs.
    """
    epochs = range(1, len(val_accuracies) + 1)

    fig, ax1 = plt.subplots()

    # Plot validation accuracy
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Validation Accuracy (%)', color='tab:blue')
    acc_line, = ax1.plot(epochs, val_accuracies, color='tab:blue', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a second y-axis for validation loss
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Loss', color='tab:red')
    loss_line, = ax2.plot(epochs, val_losses, color='tab:red', label='Loss')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # # Combine legends from both axes
    # fig.legend([acc_line, loss_line], ['Accuracy', 'Loss'], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    # Title and grid
    plt.title('Validation Accuracy and Loss Over Epochs')
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the legend
    plt.show()

def collect_misclassified_images(model, dataloader, output_dir, device):
    """
    Collect misclassified images and save them in folders based on their true labels,
    along with their predicted labels and confidence scores. Additionally, compute statistics
    about the misclassified and total images per class.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        dataloader (torch.utils.data.DataLoader): The DataLoader containing the dataset.
        output_dir (str): The directory where misclassified images will be saved.
        device (torch.device): The device (CPU or GPU) to use for inference.

    Returns:
        dict: A dictionary containing counts of total and misclassified images per class.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Prepare the output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Remove existing directory
    os.makedirs(output_dir)

    total_counts = Counter()  # Total images per class
    misclassified_counts = Counter()  # Misclassified images per class

    # Disable gradient computation
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            # Update total counts for each class
            total_counts.update(labels.cpu().tolist())

            # Identify misclassified indices
            misclassified_indices = (predictions != labels).nonzero(as_tuple=True)[0]

            for idx in misclassified_indices:
                image = images[idx]
                true_label = labels[idx].item()
                predicted_label = predictions[idx].item()
                confidence = torch.softmax(outputs[idx], 0)[predicted_label].item()  # Confidence score

                # Update misclassified counts
                misclassified_counts[true_label] += 1

                # Create subdirectory for the true label
                true_label_dir = os.path.join(output_dir, f"class_{true_label}")
                if not os.path.exists(true_label_dir):
                    os.makedirs(true_label_dir)

                # Save the misclassified image
                image_path = os.path.join(
                    true_label_dir, f"img_{batch_idx}_{idx}_pred_{predicted_label}_conf_{confidence:.2f}.png"
                )
                save_image(image, image_path)

    # Print statistics
    print("Class-wise statistics:")
    for cls in sorted(total_counts.keys()):
        total = total_counts[cls]
        misclassified = misclassified_counts[cls]
        error_rate = (misclassified / total) * 100 if total > 0 else 0
        print(f"{cls}: Total={total}, Misclassified={misclassified}, Error Rate={error_rate:.2f}%")

    # Return statistics for further analysis
    return {
        "total_counts": total_counts,
        "misclassified_counts": misclassified_counts
    }

def plot_class_statistics(stats):
    """
    Plot the statistics of total and misclassified images per class.

    Args:
        stats (dict): Dictionary containing 'total_counts' and 'misclassified_counts'.
    """
    total_counts = stats['total_counts']
    misclassified_counts = stats['misclassified_counts']

    # Convert counters to sorted lists
    classes = sorted(total_counts.keys())
    total = [total_counts[cls] for cls in classes]
    misclassified = [misclassified_counts[cls] for cls in classes]
    error_rate = [(misclassified[i] / total[i]) * 100 if total[i] > 0 else 0 for i in range(len(classes))]

    # Create bar plot
    x = np.arange(len(classes))  # Class indices

    plt.figure(figsize=(10, 6))

    # Bar plots for total and misclassified images
    plt.bar(x - 0.2, total, width=0.4, label='Total Images', color='skyblue')
    plt.bar(x + 0.2, misclassified, width=0.4, label='Misclassified Images', color='salmon')

    # Add error rate as a line plot
    plt.plot(x, error_rate, label='Error Rate (%)', color='green', marker='o', linewidth=2)

    # Add labels and title
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count / Error Rate (%)', fontsize=12)
    plt.title('Class-wise Statistics', fontsize=14)
    plt.xticks(x, [f'{cls}' for cls in classes], rotation=45, fontsize=10)
    plt.legend()
    plt.tight_layout()

    # Show plot
    plt.show()

# Example usage:
# Assuming `stats` is the dictionary returned by `collect_misclassified_images`
# plot_class_statistics(stats)

def plot_high_error_classes(stats):
    """
    Plot statistics for classes with an error rate higher than the average error rate.

    Args:
        stats (dict): Dictionary containing 'total_counts' and 'misclassified_counts'.
    """
    total_counts = stats['total_counts']
    misclassified_counts = stats['misclassified_counts']

    # Convert counters to sorted lists
    classes = sorted(total_counts.keys())
    total = [total_counts[cls] for cls in classes]
    misclassified = [misclassified_counts[cls] for cls in classes]
    error_rate = [(misclassified[i] / total[i]) * 100 if total[i] > 0 else 0 for i in range(len(classes))]

    # Calculate average error rate
    avg_error_rate = sum(error_rate) / len([e for e in error_rate if e > 0])

    # Filter classes with error rate above average
    filtered_data = [(cls, total[i], misclassified[i], error_rate[i])
                     for i, cls in enumerate(classes) if error_rate[i] > avg_error_rate]

    # Unpack filtered data
    filtered_classes, filtered_total, filtered_misclassified, filtered_error_rate = zip(*filtered_data)

    # Create bar plot
    x = np.arange(len(filtered_classes))  # Class indices

    plt.figure(figsize=(10, 6))

    # Bar plots for total and misclassified images
    plt.bar(x - 0.2, filtered_total, width=0.4, label='Total Images', color='skyblue')
    plt.bar(x + 0.2, filtered_misclassified, width=0.4, label='Misclassified Images', color='salmon')

    # Add error rate as a line plot
    plt.plot(x, filtered_error_rate, label='Error Rate (%)', color='green', marker='o', linewidth=2)

    # Add labels and title
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count / Error Rate (%)', fontsize=12)
    plt.title(f'Classes with Error Rate Above Average ({avg_error_rate:.2f}%)', fontsize=14)
    plt.xticks(x, [f'{cls}' for cls in filtered_classes], rotation=45, fontsize=10)
    plt.legend()
    plt.tight_layout()

    # Show plot
    plt.show()

# Example usage:
# Assuming `stats` is the dictionary returned by `collect_misclassified_images`
# plot_high_error_classes(stats)

