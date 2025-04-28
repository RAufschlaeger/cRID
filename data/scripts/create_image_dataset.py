# encoding: utf-8
"""
@author:  raufschlaeger
"""

import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from data.datasets import ImageDataset

import logging

logging.getLogger().setLevel(logging.ERROR)

# Ensure the tools directory is also in the Python path
tools_path = os.path.join(project_root, 'tools')
if tools_path not in sys.path:
    sys.path.append(tools_path)

import torch  # Add import for torch
import csv  # Add import for csv module

from tools.config import cfg  # Import the configuration object
    
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def create_image_dataset():
    """
    Creates a dataset for image data using the provided configuration.

    Returns:
        Dataset: A PyTorch Dataset object for image data.
    """
    transform = transforms.Compose([
        transforms.Resize((cfg.MODEL.IMAGE_SIZE, cfg.MODEL.IMAGE_SIZE)),  # Use IMAGE_SIZE from config
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.DATASETS.MEAN, std=cfg.DATASETS.STD)
    ])
    dataset = ImageDataset(
        root_dir=cfg.DATASETS.ROOT_DIR,
        annotations_file=cfg.DATASETS.ANNOTATIONS_FILE,
        transform=transform
    )
    print(f"Dataset size: {len(dataset)}")  # Print the size of the dataset
    return dataset


if __name__ == '__main__':
    # Example loop structure for processing multiple models, datasets, and splits
    for MODEL_NAME in ['llama32vision', 'allenai-Molmo-7B-O-0924']:
        for DATASET in ["Market-1501-v15.09.15", "cuhk03-np/labeled", "cuhk03-np/detected"]:
            for split in ["bounding_box_test", "bounding_box_train", "query"]:
                try:
                    # Example configuration setup
                    cfg.DATASETS.ROOT_DIR = os.path.join(project_root, f"data/raw/{DATASET}/{split}")
                    
                    # Check for model-specific annotation file
                    model_specific_file = os.path.join(project_root, f"data/raw/{DATASET}/{split}/annotations_{MODEL_NAME}.csv")                    
                    if os.path.exists(model_specific_file):
                        cfg.DATASETS.ANNOTATIONS_FILE = model_specific_file
                        print(model_specific_file)
                    else:
                        print(f"Annotations file not found: {model_specific_file}")
                        continue
                    
                    print(f"Using annotations file: {cfg.DATASETS.ANNOTATIONS_FILE}")
                    
                    # Create the dataset
                    dataset = create_image_dataset()
                    print(f"Dataset created for MODEL: {MODEL_NAME}, DATASET: {DATASET}, SPLIT: {split}")
                    
                    # Save the dataset as a .pth file
                    output_path = os.path.join(project_root, f"data/images/{DATASET}/{split}_image_dataset_{MODEL_NAME}.pth")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    torch.save(dataset, output_path)
                    print(f"Dataset saved to {output_path}")
                    
                    # # Load the saved dataset and print an example item
                    # loaded_dataset = torch.load(output_path, weights_only=False)  # Set weights_only=True
                    # example_item = loaded_dataset[0]  # Get the first item
                    # print(f"Example item from {output_path}: {example_item}")
                except Exception as e:
                    print(f"Failed to process MODEL: {MODEL_NAME}, DATASET: {DATASET}, SPLIT: {split}. Error: {str(e)}")
