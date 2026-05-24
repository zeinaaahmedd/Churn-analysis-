import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_validate_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    if 'Churn' not in df.columns:
        raise ValueError("Validation Failed: 'Churn' target column is missing.")
        
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        print(f"Warning: Found {null_count} unexpected missing values. Dropping them.")
        df = df.dropna()
        
    distribution = df['Churn'].value_counts(normalize=True).to_dict()
    print(f"Original Target Distribution: {distribution}")
    return df

def preprocess_and_split_data(df):
    # Drop unneeded identifier or raw text columns
    cols_to_drop = ['Unnamed: 0', 'customerID', 'PromptInput', 'CustomerFeedback']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Separate numeric and categorical columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # one encoder per column so each can be saved and reused at inference time
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        
    # --- STEP 1: Split into Train (70%) and Temporary (30%) ---
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    # --- STEP 2: Split Temporary (30%) equally into Val (15%) and Test (15%) ---
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    # Feature Scaling (Fit on Train, transform on Val and Test to avoid data leakage)
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    print(f"\nFinal Splits:")
    print(f"Train Shape: {X_train.shape} | Val Shape: {X_val.shape} | Test Shape: {X_test.shape}")
    print(f"Train Class Dist: {y_train.value_counts(normalize=True).to_dict()}")
    print(f"Val Class Dist: {y_val.value_counts(normalize=True).to_dict()}")
    print(f"Test Class Dist: {y_test.value_counts(normalize=True).to_dict()}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoders, scaler, X_train.columns.tolist()

def main():
    raw_data_path = "data/raw/telco_prep.csv"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    df = load_and_validate_data(raw_data_path)
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoders, scaler, feature_cols = preprocess_and_split_data(df)

    # Save processed splits
    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(processed_dir, "X_val.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(processed_dir, "y_val.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

    # Save preprocessor artifacts for the inference service
    os.makedirs("models", exist_ok=True)
    joblib.dump(label_encoders, "models/label_encoders.pkl")
    joblib.dump(scaler,         "models/scaler.pkl")
    joblib.dump(feature_cols,   "models/feature_columns.pkl")

    print("\nData preparation complete! Train, Validation, and Test files saved.")
    print("Preprocessor artifacts saved to models/")

if __name__ == "__main__":
    main()