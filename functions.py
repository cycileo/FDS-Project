import os
import random
import shutil
from pathlib import Path
import gdown
import zipfile

def split_validation_set(val_dir, test_dir, split_ratio=0.5, seed=42):
    """
    Splits the validation dataset into new validation and test sets.
    
    Args:
        val_dir (str): Path to the current validation directory.
        test_dir (str): Path to the test directory to be created.
        split_ratio (float): Ratio of samples to move to the test set (default 0.5).
        seed (int): Random seed for reproducibility.
    """
    random.seed(seed)
    
    # Create the test directory if it doesn't exist
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    
    # Iterate over each class folder in the validation directory
    for class_name in os.listdir(val_dir):
        class_val_dir = os.path.join(val_dir, class_name)
        class_test_dir = os.path.join(test_dir, class_name)
        
        # Create corresponding class directory in test set
        Path(class_test_dir).mkdir(parents=True, exist_ok=True)
        
        # Get all files in the class directory
        files = os.listdir(class_val_dir)
        files = [f for f in files if os.path.isfile(os.path.join(class_val_dir, f))]
        
        # Shuffle files and split
        random.shuffle(files)
        split_idx = int(len(files) * split_ratio)
        test_files = files[:split_idx]
        
        # Move test files to the test directory
        for file_name in test_files:
            src_path = os.path.join(class_val_dir, file_name)
            dest_path = os.path.join(class_test_dir, file_name)
            shutil.move(src_path, dest_path)
        
        print(f"Class '{class_name}': Moved {len(test_files)} files to test set.")

    print("Validation set successfully split into new validation and test sets.")





def download_and_extract_zip(url, extract_to='PlantVillage'):
    """
    Downloads a zip file from Google Drive and extracts it to a specified directory.

    This function downloads a zip file from Google Drive using the provided URL, extracts its contents into the specified
    directory, and then deletes the zip file.

    Args:
        url (str): URL of the zip file on Google Drive. It should be in the format 'https://drive.google.com/file/d/{file_id}/view'.
        extract_to (str): Directory where the contents of the zip file will be extracted (default is 'PlantVillage').

    Returns:
        None
    """
    # Generate the direct download URL for the file
    file_id = url.split('/d/')[1].split('/')[0]
    download_url = f'https://drive.google.com/uc?id={file_id}'

    # Download the zip file
    zip_file = 'PlantVillage(split).zip'
    gdown.download(download_url, zip_file, quiet=False)

    # Extract the zip file directly
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Extract all the files directly into the 'extract_to' folder
        zip_ref.extractall(extract_to)

    # Remove the zip file after extraction
    os.remove(zip_file)


