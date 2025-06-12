# encoding: utf-8
"""
@author:  raufschlaeger
"""

import os
import sys
from datetime import datetime

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import pandas as pd
import torch
from datasets import tqdm
import logging
from data.src.models.molmo7b import Molmo
from data.src.models.internvl3_8b import InternVL3


def setup_logging(split: str):
    # Create a timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create absolute path to logs directory within project structure
    logs_dir = os.path.join(project_root, 'data', 'logs')
    
    # Ensure logs directory exists
    os.makedirs(logs_dir, exist_ok=True)
    
    log_filename = os.path.join(logs_dir, f"create_dataset_{split}_{timestamp}.txt")

    # Configure logging to write to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_filename


def update_csv(idx, graph, annotations, annotations_file):
    """
    Updates the caption for a single entry (specified by idx) in the CSV file.

    Args:
        idx (int): Index of the row to update.
        graph (str): The graph to add.
        annotations (pd.DataFrame): The annotations DataFrame.
        annotations_file (str): Path to save the updated CSV file.
    """
    try:
        annotations.at[idx, 'graph'] = graph
        annotations.to_csv(annotations_file, index=False)
        logging.info(f"Updated caption for index {idx} and saved to {annotations_file}")
    except Exception as e:
        logging.error(f"Failed to update CSV for index {idx}: {e}")


def main(model_name: str, dataset: str, split: str, model_safe_name: str):

    # Setup logging
    log_filename = setup_logging(split)
    logging.info(f"Starting processing for split: {split}")
    torch.cuda.empty_cache()

    try:
        # Create absolute paths using project_root
        raw_dir = os.path.join(project_root, 'data')
        annotations_dir = os.path.join(project_root, 'data', 'annotations')
        annotations_file = os.path.join(annotations_dir, dataset, split, f"annotations_{model_safe_name}.csv")
        print(annotations_file)
        img_dir = os.path.join(raw_dir, dataset, split)
        
        # Check if paths exist
        if not os.path.exists(annotations_file):
            logging.error(f"Annotation file does not exist: {annotations_file}")
            raise FileNotFoundError(f"Annotation file not found: {annotations_file}")
            
        if not os.path.isdir(img_dir):
            logging.error(f"Image directory does not exist: {img_dir}")
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        annotations = pd.read_csv(annotations_file)
        logging.info(f"Loaded annotations from {annotations_file}")

        if "Molmo-7B-O-0924" in model_safe_name:
            extractor = Molmo()
        elif "InternVL3" in model_safe_name:
            extractor = InternVL3()
        # elif "llama32vision" in model_safe_name:
        #     extractor = Llama()
        else:
            raise ValueError(f"Model {model_name} not supported")

        start = 0
        end = len(annotations)

        for idx in tqdm(range(start, end), desc=f"Processing {split}"):
            image_name = annotations.iloc[idx, 0]
            image_path = os.path.join(img_dir, image_name)

            try:
                graph = extractor.process_image(image_path)
                logging.debug(f"Processed image {image_path} - Graph: {graph}")

                update_csv(idx, graph, annotations, annotations_file)

            except Exception as e:
                logging.error(f"Error processing image {image_path} at index {idx}: {e}")

    except Exception as e:
        logging.error(f"Failed to complete processing for split {split}: {e}")
        raise

    logging.info("Processing completed")
    return log_filename

# tmux new -s s1
# tmux attach -t s1

# s1
# if __name__ == '__main__':
#     for MODEL_NAME in ['allenai/Molmo-7B-O-0924']:
#         model_safe_name = MODEL_NAME.replace("/", "-")
#         print(f"Processing model: {MODEL_NAME}")    
#         for split in ["query", "bounding_box_train"]:
#             for DATASET in ["Market-1501-v15.09.15"]:
#                 try:
#                     log_file = main(MODEL_NAME, DATASET, split=split, model_safe_name=model_safe_name)
#                     print(f"\nLog file created at: {log_file}")
#                 except Exception as e:
#                     print(f"Failed to complete processing: {str(e)}")

# s2
# if __name__ == '__main__':
#     for MODEL_NAME in ['allenai/Molmo-7B-O-0924']:
#         model_safe_name = MODEL_NAME.replace("/", "-")
#         print(f"Processing model: {MODEL_NAME}")    
#         for split in ["bounding_box_test"]:
#             for DATASET in ["Market-1501-v15.09.15"]:
#                 try:
#                     log_file = main(MODEL_NAME, DATASET, split=split, model_safe_name=model_safe_name)
#                     print(f"\nLog file created at: {log_file}")
#                 except Exception as e:
#                     print(f"Failed to complete processing: {str(e)}")

# s3
# if __name__ == '__main__':
#     for MODEL_NAME in ['allenai/Molmo-7B-O-0924']:
#         model_safe_name = MODEL_NAME.replace("/", "-")
#         print(f"Processing model: {MODEL_NAME}")    
#         for split in ["query", "bounding_box_train"]:
#             for DATASET in ["cuhk03-np/labeled"]:
#                 try:
#                     log_file = main(MODEL_NAME, DATASET, split=split, model_safe_name=model_safe_name)
#                     print(f"\nLog file created at: {log_file}")
#                 except Exception as e:
#                     print(f"Failed to complete processing: {str(e)}")

# s4
# if __name__ == '__main__':
#     for MODEL_NAME in ['allenai/Molmo-7B-O-0924']:
#         model_safe_name = MODEL_NAME.replace("/", "-")
#         print(f"Processing model: {MODEL_NAME}")    
#         for split in ["bounding_box_test"]:
#             for DATASET in ["cuhk03-np/labeled"]:
#                 try:
#                     log_file = main(MODEL_NAME, DATASET, split=split, model_safe_name=model_safe_name)
#                     print(f"\nLog file created at: {log_file}")
#                 except Exception as e:
#                     print(f"Failed to complete processing: {str(e)}")

# s5
# if __name__ == '__main__':
#     for MODEL_NAME in ['allenai/Molmo-7B-O-0924']:
#         model_safe_name = MODEL_NAME.replace("/", "-")
#         print(f"Processing model: {MODEL_NAME}")    
#         for split in ["query", "bounding_box_train"]:
#             for DATASET in ["cuhk03-np/detected"]:
#                 try:
#                     log_file = main(MODEL_NAME, DATASET, split=split, model_safe_name=model_safe_name)
#                     print(f"\nLog file created at: {log_file}")
#                 except Exception as e:
#                     print(f"Failed to complete processing: {str(e)}")

# s6
# if __name__ == '__main__':
#     for MODEL_NAME in ['allenai/Molmo-7B-O-0924']:
#         model_safe_name = MODEL_NAME.replace("/", "-")
#         print(f"Processing model: {MODEL_NAME}")    
#         for split in ["bounding_box_test"]:
#             for DATASET in ["cuhk03-np/detected"]:
#                 try:
#                     log_file = main(MODEL_NAME, DATASET, split=split, model_safe_name=model_safe_name)
#                     print(f"\nLog file created at: {log_file}")
#                 except Exception as e:
#                     print(f"Failed to complete processing: {str(e)}")

# s1-s6 one after another
if __name__ == '__main__':
    for MODEL_NAME in ['OpenGVLab/InternVL3-8B']:
        model_safe_name = MODEL_NAME.replace("/", "-")
        print(f"Processing model: {MODEL_NAME}")    
        for split in ["query", "bounding_box_test", "bounding_box_train"]:
            for DATASET in ["Market-1501-v15.09.15", "cuhk03-np/labeled", "cuhk03-np/detected"]:
                try:
                    log_file = main(MODEL_NAME, DATASET, split=split, model_safe_name=model_safe_name)
                    print(f"\nLog file created at: {log_file}")
                except Exception as e:
                    print(f"Failed to complete processing: {str(e)}")
