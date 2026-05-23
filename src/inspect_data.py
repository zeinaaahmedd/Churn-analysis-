import pandas as pd
import os

# Define paths
RAW_DATA_DIR = "data/raw"
files = [
    "telco_prep.csv",
    "telco_churn_with_all_feedback.csv",
    "telco_noisy_feedback_prep.csv"
]

for file_name in files:
    file_path = os.path.join(RAW_DATA_DIR, file_name)
    if os.path.exists(file_path):
        print(f"\n=================== Analyzing: {file_name} ===================")
        df = pd.read_csv(file_path)
        
        # 1. Shape and Columns
        print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print("\nColumns and Data Types:")
        print(df.dtypes)
        
        # 2. Basic Null Check
        null_counts = df.isnull().sum()
        missing_data = null_counts[null_counts > 0]
        if not missing_data.empty:
            print("\nMissing Values:")
            print(missing_data)
        else:
            print("\nNo missing values found!")
            
        # 3. Target Distribution Check
        # Let's see if 'Churn' or a similar target column is in this specific file
        target_col = [col for col in df.columns if 'churn' in col.lower() or 'target' in col.lower()]
        if target_col:
            print(f"\nTarget Distribution ({target_col[0]}):")
            print(df[target_col[0]].value_counts(normalize=True) * 100)
        else:
            print("\nNo explicit Churn/Target column found by name in this file.")
    else:
        print(f"File not found: {file_path}")