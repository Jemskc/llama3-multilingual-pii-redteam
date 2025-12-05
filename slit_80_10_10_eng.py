import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = "Manually-checked-spanish-data.xlsx"  # Your exact file name
TARGET_TOTAL = 500                         # Strict limit
RANDOM_SEED = 42

def split_excel_data(input_path):
    # 1. Check if file exists
    if not os.path.exists(input_path):
        print(f"Error: The file '{input_path}' was not found.")
        return

    print(f"Reading Excel file: {input_path}...")
    
    # 2. Load Excel File (Engine 'openpyxl' is required for .xlsx)
    try:
        df = pd.read_excel(input_path, engine='openpyxl')
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    current_rows = len(df)
    print(f"Loaded {current_rows} rows successfully.")

    # 3. Strict Truncation (Cap at 500)
    if current_rows > TARGET_TOTAL:
        print(f"Truncating dataset from {current_rows} to {TARGET_TOTAL} rows.")
        df = df.iloc[:TARGET_TOTAL]
    elif current_rows < TARGET_TOTAL:
        print(f"Warning: Dataset has only {current_rows} rows. (Less than 500). Using all available.")

    # 4. Perform the Split (80% Train, 10% Val, 10% Test)
    # First split: 80% Train, 20% Temp
    train_data, temp_data = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, shuffle=True
    )
    # Second split: Split the 20% into two equal halves (10% Val, 10% Test)
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=RANDOM_SEED, shuffle=True
    )

    # 5. Save as CSV (UTF-8)
    # We save as CSV so the benchmarking script can read it easily as text later.
    train_data.to_csv("spanish_train.csv", index=False, encoding='utf-8')
    val_data.to_csv("spanish_val.csv", index=False, encoding='utf-8')
    test_data.to_csv("spanish_test.csv", index=False, encoding='utf-8')

    print("-" * 30)
    print("SUCCESS: Files converted to CSV and Split")
    print("-" * 30)
    print(f"1. spanish_train.csv: {len(train_data)} rows")
    print(f"2. spanish_val.csv:   {len(val_data)} rows")
    print(f"3. spanish_test.csv:  {len(test_data)} rows")
    print("-" * 30)

if __name__ == "__main__":
    split_excel_data(INPUT_FILE)
