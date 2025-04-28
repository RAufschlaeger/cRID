# encoding: utf-8
"""
@author:  raufschlaeger
"""

import os
import csv

def parse_filename_market1501(filename):
    parts = filename.split('_')
    person_id = parts[0]
    camera_id = parts[1][1]  # Remove the 'c' prefix
    sequence_number = parts[2][1]  # Remove the 's' prefix
    frame_number = parts[3].split('.')[0]  # Remove the file extension
    return person_id, camera_id, sequence_number, frame_number

def parse_filename_cuhk03(filename):
    parts = filename.split('_')
    person_id = parts[0]
    camera_id = parts[1][1]
    frame_number = parts[2].split('.')[0]  # Remove the file extension
    return person_id, camera_id, frame_number


def create_annotations_csv(folder_path, output_csv, DATASET):
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist. Creating it.")
        os.makedirs(folder_path, exist_ok=True)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
    print(f"Found {len(image_files)} images in {folder_path}")
    
    if not image_files:
        print(f"Warning: No images found in {folder_path}")
        # Create an empty CSV file anyway
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            if DATASET == "Market-1501-v15.09.15":
                writer.writerow(['filename', 'ID', 'camera ID', 'sequence number', 'frame number', 'graph', 'fixed_graph'])
            elif DATASET == "cuhk03-np/labeled" or DATASET == "cuhk03-np/detected":
                writer.writerow(['filename', 'ID', 'camera ID', 'frame number', 'graph', 'fixed_graph'])
        print(f"Created empty CSV file: {output_csv}")
        return

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        if DATASET == "Market-1501-v15.09.15":
            writer.writerow(['filename', 'ID', 'camera ID', 'sequence number', 'frame number', 'graph', 'fixed_graph'])
        elif DATASET == "cuhk03-np/labeled" or DATASET == "cuhk03-np/detected":
            writer.writerow(['filename', 'ID', 'camera ID', 'frame number', 'graph', 'fixed_graph'])
        
        rows_written = 0
        for filename in image_files:
            try:
                if DATASET == "Market-1501-v15.09.15":
                    person_id, camera_id, sequence_number, frame_number = parse_filename_market1501(filename)
                    writer.writerow([filename, person_id, camera_id, sequence_number, frame_number])
                elif DATASET == "cuhk03-np/labeled" or DATASET == "cuhk03-np/detected":
                    person_id, camera_id, frame_number = parse_filename_cuhk03(filename)
                    writer.writerow([filename, person_id, camera_id, frame_number])
                rows_written += 1
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
        
        print(f"Wrote {rows_written} rows to {output_csv}")


def main():
    # Use absolute paths instead of relative paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'annotations'))
    print(f"Base directory: {base_dir}")
    
    for MODEL_NAME in ['allenai/Molmo-7B-O-0924']:
        model_safe_name = MODEL_NAME.replace("/", "-")
        print(f"Processing model: {MODEL_NAME}")
        
        for DATASET in ["cuhk03-np/labeled", "cuhk03-np/detected", "Market-1501-v15.09.15"]:
            print(f"  Processing dataset: {DATASET}")
            
            for split in ['bounding_box_train', 'bounding_box_test', 'query']:
                print(f"    Processing split: {split}")
                folder_path = os.path.join(base_dir, DATASET, split)
                output_csv = os.path.join(folder_path, f'annotations_{model_safe_name}.csv')
                
                print(f"    Source folder: {folder_path}")
                print(f"    Output CSV: {output_csv}")
                
                create_annotations_csv(folder_path, output_csv, DATASET)
                print(f"    Completed processing {split}")


if __name__ == "__main__":
    main()
