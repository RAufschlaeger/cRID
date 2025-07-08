import pandas as pd

def count_nan_in_last_column(file_path):
    """
    Counts how often NaN appears in the last column of a CSV file and returns
    both the NaN count and the total number of rows.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing (nan_count, total_rows).
               Returns (-1, -1) if an error occurs or file is empty.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)

        # Get the total number of rows
        total_rows = len(df) # Or df.shape[0]

        # Check if the DataFrame is empty
        if df.empty:
            print(f"The CSV file '{file_path}' is empty.")
            return (0, 0) # Return 0 for both if empty

        # Get the name of the last column
        last_column_name = df.columns[-1]

        # Select the last column and count NaN values
        nan_count = df[last_column_name].isna().sum()

        return (nan_count, total_rows)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return (-1, -1)
    except Exception as e:
        print(f"An error occurred: {e}")
        return (-1, -1)

# --- Example Usage with Direct File Path ---
if __name__ == "__main__":
    # Create a dummy CSV file for testing
    dummy_data = {
        'col1': [1, 2, 3, 4, 5],
        'col2': ['A', 'B', 'C', 'D', 'E'],
        'col3': [10.1, float('nan'), 12.3, float('nan'), 14.5]
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df.to_csv('your_test_file.csv', index=False) # Save the dummy DataFrame to a CSV

    # Directly specify the file path
    file_to_check = 'your_test_file.csv'
    file_to_check = 'data/annotations/Market-1501-v15.09.15/bounding_box_train/annotations_allenai-Molmo-7B-O-0924.csv'
    # Number of NaN values in the last column of 'data/annotations/Market-1501-v15.09.15/bounding_box_train/annotations_allenai-Molmo-7B-O-0924.csv': (12927, 12936)
    file_to_check = 'data/annotations/Market-1501-v15.09.15/bounding_box_test/annotations_allenai-Molmo-7B-O-0924.csv'
    # Number of NaN values in the last column of 'data/annotations/Market-1501-v15.09.15/bounding_box_test/annotations_allenai-Molmo-7B-O-0924.csv': (19715, 19732)
    file_to_check = 'data/annotations/Market-1501-v15.09.15/query/annotations_allenai-Molmo-7B-O-0924.csv'
    # Number of NaN values in the last column of 'data/annotations/Market-1501-v15.09.15/query/annotations_allenai-Molmo-7B-O-0924.csv': (3364, 3368)

    file_to_check = 'data/annotations/cuhk03-np/detected/bounding_box_train/annotations_allenai-Molmo-7B-O-0924.csv'
    file_to_check = 'data/annotations/cuhk03-np/detected/bounding_box_test/annotations_allenai-Molmo-7B-O-0924.csv'
    file_to_check = 'data/annotations/cuhk03-np/detected/query/annotations_allenai-Molmo-7B-O-0924.csv'

    file_to_check = 'data/annotations/cuhk03-np/labeled/bounding_box_train/annotations_allenai-Molmo-7B-O-0924.csv'
    file_to_check = 'data/annotations/cuhk03-np/labeled/bounding_box_test/annotations_allenai-Molmo-7B-O-0924.csv'
    file_to_check = 'data/annotations/cuhk03-np/labeled/query/annotations_allenai-Molmo-7B-O-0924.csv'

    # Number of NaN values in the last column of 'data/annotations/cuhk03-np/detected/bounding_box_train/annotations_allenai-Molmo-7B-O-0924.csv': (7363, 7365)
    # Number of NaN values in the last column of 'data/annotations/cuhk03-np/detected/bounding_box_test/annotations_allenai-Molmo-7B-O-0924.csv': (5332, 5332)
    # Number of NaN values in the last column of 'data/annotations/cuhk03-np/detected/query/annotations_allenai-Molmo-7B-O-0924.csv': (1399, 1400)
        
    # Number of NaN values in the last column of 'data/annotations/cuhk03-np/labeled/bounding_box_train/annotations_allenai-Molmo-7B-O-0924.csv': (7366, 7368)
    # Number of NaN values in the last column of 'data/annotations/cuhk03-np/labeled/bounding_box_test/annotations_allenai-Molmo-7B-O-0924.csv': (5325, 5328)
    # Number of NaN values in the last column of 'data/annotations/cuhk03-np/labeled/query/annotations_allenai-Molmo-7B-O-0924.csv': (1399, 1400)

    # market1501: 30 / 36036
    # cuhk03-np/detected: 3 / 14097
    # cuhk03-np/labeled: 6 / 14096


    nan_occurrences = count_nan_in_last_column(file_to_check)

    if nan_occurrences != -1:
        print(f"Number of NaN values in the last column of '{file_to_check}': {nan_occurrences}")

    print("\n--- End of Direct File Path Example ---")
    