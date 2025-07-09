import pandas as pd
import os

def modify_third_column_in_csv(
    input_csv_path: str = "all_data.csv",
    output_csv_path: str = "modified_all_data.csv"
) -> None:
    """
    Reads a CSV file, modifies the third column:
    If a value in the third column is greater than 1, it divides that value by 100.
    Otherwise, the value remains unchanged.
    Saves the modified DataFrame to a new CSV file.

    Args:
        input_csv_path (str): The path to the input CSV file (e.g., 'all_data.csv').
        output_csv_path (str): The path where the modified CSV file will be saved.
    """
    if not os.path.exists(input_csv_path):
        print(f"Error: Input CSV file not found at '{input_csv_path}'")
        return

    try:
        # Read the CSV file into a pandas DataFrame.
        # pandas automatically handles headers.
        df = pd.read_csv(input_csv_path)
        print(f"Successfully loaded data from '{input_csv_path}' with {len(df)} rows and {len(df.columns)} columns.")

        # Check if the DataFrame has at least 3 columns (0, 1, 2)
        if len(df.columns) < 3:
            print(f"Error: The CSV file '{input_csv_path}' does not have a third column to modify.")
            print(f"It only has {len(df.columns)} columns.")
            return

        # Get the name of the third column (index 2)
        third_column_name = df.columns[2]
        print(f"Targeting the third column: '{third_column_name}'")

        # Apply the conditional logic using a vectorized operation for efficiency.
        # We use .loc to ensure we are modifying the DataFrame directly and avoid SettingWithCopyWarning.
        # This checks if the value is numeric before comparison to prevent errors with non-numeric data.
        # For non-numeric values, the condition (df[third_column_name] > 1) will evaluate to False
        # or raise an error if the column contains mixed types that cannot be compared.
        # We assume the third column contains numeric data or can be coerced.
        # If there are non-numeric values, you might need more robust error handling or type conversion.
        
        # Convert the column to numeric, coercing errors to NaN
        df[third_column_name] = pd.to_numeric(df[third_column_name], errors='coerce')

        # Apply the condition only to non-NaN numeric values
        condition = (df[third_column_name] > 2) & (df[third_column_name].notna())
        df.loc[condition, third_column_name] = (df.loc[condition, third_column_name] / 100).round(3)

        print(f"Successfully modified the third column based on the condition.")

        # Save the modified DataFrame to a new CSV file.
        # index=False prevents pandas from writing the DataFrame index as a column in the CSV.
        # header=True (default) ensures the column names are written as the first row.
        df.to_csv(output_csv_path, index=False)
        print(f"Modified data saved to '{output_csv_path}'.")

    except pd.errors.EmptyDataError:
        print(f"Error: The CSV file '{input_csv_path}' is empty.")
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file '{input_csv_path}': {e}")
    except KeyError:
        print(f"Error: Column '{third_column_name}' not found after initial check (should not happen).")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy CSV file for demonstration if 'all_data.csv' doesn't exist
    # This part is for testing purposes. In a real scenario, you'd have your 'all_data.csv'.

    # Call the function to modify the third column of 'all_data.csv'
    modify_third_column_in_csv(
        input_csv_path="all_data.csv",
        output_csv_path="modified_all_data.csv"
    )