import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import pandas as pd
from tqdm import tqdm

from data.src.data.graph_dataset import GraphDataset

def main(MODEL_NAME: str, DATASET: str, split: str):
    torch.cuda.empty_cache()

    print(f"./data/annotations/{DATASET}/{split}/annotations_{MODEL_NAME}.csv")

    dataset = GraphDataset(
        annotations_file=f"./data/annotations/{DATASET}/{split}/annotations_{MODEL_NAME}.csv"
    )

    all_data = []
    start = 0
    end = len(dataset)

    annotations_path = f"./data/annotations/{DATASET}/{split}/annotations_{MODEL_NAME}.csv"
    annotations_df = pd.read_csv(annotations_path)

    annotations_df["fixed_graph"] = annotations_df["fixed_graph"].astype(str)  # Explicitly cast column to string

    for i in tqdm(range(start, end), desc="Processing dataset"):
        try:
            sample, label, fixed_graph = dataset[i]
            all_data.append((sample, label))
            if fixed_graph is not None:
                annotations_df.loc[i, "fixed_graph"] = str(fixed_graph)  # Explicitly cast to string
        except Exception as e:
            print(f"Error processing MODEL_NAME: {MODEL_NAME}, DATASET: {DATASET}, split: {split}, index: {i}")
            print(f"Exception: {str(e)}")

    # Save updated annotations file
    annotations_df.to_csv(annotations_path, index=False)

    output_path = f"./data/graphs/{DATASET}/{split}/graph_dataset_{MODEL_NAME}.pth"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure parent directory exists
    torch.save(all_data, output_path)
    print(f"Dataset saved to {output_path}")

    return None

if __name__ == '__main__':
    for MODEL_NAME in ['allenai-Molmo-7B-O-0924']:
        for DATASET in ["Market-1501-v15.09.15", "cuhk03-np/detected", "cuhk03-np/labeled"]:
            for split in ["bounding_box_test", "bounding_box_train", "query"]:
                try:
                    log_file = main(MODEL_NAME, DATASET, split)
                    print(f"\nLog file created at: {log_file}")
                except Exception as e:
                    print(f"Failed to complete processing: {str(e)}")

# Uncomment the desired combination to run
# try:
#     log_file = main('allenai-Molmo-7B-O-0924', "Market-1501-v15.09.15", "query")
#     print(f"\nLog file created at: {log_file}")
# except Exception as e:
#     print(f"Failed to complete processing: {str(e)}")

# try:
#     log_file = main('allenai-Molmo-7B-O-0924', "Market-1501-v15.09.15", "bounding_box_test")
#     print(f"\nLog file created at: {log_file}")
# except Exception as e:
#     print(f"Failed to complete processing: {str(e)}")

# try:
#     log_file = main('allenai-Molmo-7B-O-0924', "Market-1501-v15.09.15", "bounding_box_train")
#     print(f"\nLog file created at: {log_file}")
# except Exception as e:
#     print(f"Failed to complete processing: {str(e)}")

# try:
#     log_file = main('allenai-Molmo-7B-O-0924', "cuhk03-np/detected", "query")
#     print(f"\nLog file created at: {log_file}")
# except Exception as e:
#     print(f"Failed to complete processing: {str(e)}")

# try:
#     log_file = main('allenai-Molmo-7B-O-0924', "cuhk03-np/detected", "bounding_box_test")
#     print(f"\nLog file created at: {log_file}")
# except Exception as e:
#     print(f"Failed to complete processing: {str(e)}")

# try:
#     log_file = main('allenai-Molmo-7B-O-0924', "cuhk03-np/detected", "bounding_box_train")
#     print(f"\nLog file created at: {log_file}")
# except Exception as e:
#     print(f"Failed to complete processing: {str(e)}")

# try:
#     log_file = main('allenai-Molmo-7B-O-0924', "cuhk03-np/labeled", "query")
#     print(f"\nLog file created at: {log_file}")
# except Exception as e:
#     print(f"Failed to complete processing: {str(e)}")

# try:
#     log_file = main('allenai-Molmo-7B-O-0924', "cuhk03-np/labeled", "bounding_box_test")
#     print(f"\nLog file created at: {log_file}")
# except Exception as e:
#     print(f"Failed to complete processing: {str(e)}")

# try:
#     log_file = main('allenai-Molmo-7B-O-0924', "cuhk03-np/labeled", "bounding_box_train")
#     print(f"\nLog file created at: {log_file}")
# except Exception as e:
#     print(f"Failed to complete processing: {str(e)}")
