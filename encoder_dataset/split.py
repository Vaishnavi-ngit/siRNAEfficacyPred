import pandas as pd
import os
from sklearn.model_selection import train_test_split
import random

def split_csv_into_train_dev(
    csv_file_path: str,
    train_ratio: float = 0.85,
    num_splits: int = 10,
    output_base_dir: str = "datasets"
) -> None:
    """
    Splits a given CSV file into training and validation sets multiple times
    with different random seeds and saves them to a structured directory.

    Args:
        csv_file_path (str): The path to the input CSV file.
        train_ratio (float): The ratio of the dataset to be used for training.
                             The validation ratio will be 1 - train_ratio.
                             Defaults to 0.85 (85% train, 15% validation).
        num_splits (int): The number of times to perform the split.
                          Each split will use a different random seed.
                          Defaults to 10.
        output_base_dir (str): The base directory where the split datasets
                               will be saved. Defaults to "datasets".
    """
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at '{csv_file_path}'")
        return

    try:
        # Load the CSV file. pandas.read_csv automatically detects and handles headers.
        df = pd.read_csv(csv_file_path)
        print(f"Successfully loaded data from '{csv_file_path}' with {len(df)} rows.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Create the base output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"Created base directory: '{output_base_dir}'")
    s = random.randint(0, 1000000)

    for i in range(num_splits):
        # Generate a unique random seed for each split
        seed = s+i
        print(f"\n--- Performing split {i+1}/{num_splits} with seed: {seed} ---")

        # Create a folder for the current split
        split_dir = os.path.join(output_base_dir, f"split{i+1}")
        os.makedirs(split_dir, exist_ok=True)
        print(f"Created split directory: '{split_dir}'")

        # Perform the train-validation split
        # stratify=None is used here as no specific column for stratification was mentioned.
        # If you have a target variable and want to maintain its distribution
        # across splits, you would set stratify=df['your_target_column'].
        train_df, dev_df = train_test_split(
            df,
            train_size=train_ratio,
            random_state=seed,
            shuffle=True  # Ensure data is shuffled before splitting
        )

        # Define file paths for train and dev sets
        train_output_path = os.path.join(split_dir, "train.csv")
        dev_output_path = os.path.join(split_dir, "dev.csv")

        # Save the split dataframes to CSV files.
        # By default, to_csv writes the header and does not write the DataFrame index.
        train_df.to_csv(train_output_path, index=False)
        dev_df.to_csv(dev_output_path, index=False)

        print(f"Saved training data to: '{train_output_path}' (Rows: {len(train_df)})")
        print(f"Saved validation data to: '{dev_output_path}' (Rows: {len(dev_df)})")

    print(f"\nCSV splitting process completed for {num_splits} splits.")

# --- Example Usage ---
if __name__ == "__main__":
    # Specify the path to your actual data file
    actual_csv_path = "modified_all_data.csv"

    # Call the function to split the CSV
    split_csv_into_train_dev(
        csv_file_path=actual_csv_path,
        train_ratio=0.85,
        num_splits=10,
        output_base_dir="datasets"
    )