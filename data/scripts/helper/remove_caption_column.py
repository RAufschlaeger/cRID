import os
import pandas as pd

def remove_caption_column(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path)
                    
                    # Remove the 'caption' column if it exists
                    if 'caption' in df.columns:
                        df.drop(columns=['caption'], inplace=True)
                        df.to_csv(file_path, index=False)
                        print(f"Removed 'caption' column from: {file_path}")
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

if __name__ == "__main__":
    annotations_folder = "/home/raufschlaeger/reid-strong-baseline/data/annotations"
    remove_caption_column(annotations_folder)
