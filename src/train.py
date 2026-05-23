import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_processed_data():
    processed_dir = "data/processed"
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_val = pd.read_csv(os.path.join(processed_dir, "X_val.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).values.ravel()
    y_val = pd.read_csv(os.path.join(processed_dir, "y_val.csv")).values.ravel()
    return X_train, X_val, y_train, y_val

def evaluate_model(model, X_val, y_val):
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]
    
    metrics = {
        "Accuracy": accuracy_score(y_val, preds),
        "Precision": precision_score(y_val, preds),
        "Recall": recall_score(y_val, preds),
        "F1-Score": f1_score(y_val, preds),
        "ROC-AUC": roc_auc_score(y_val, probs)
    }
    return metrics

def main():
    # Load data splits
    X_train, X_val, y_train, y_val = load_processed_data()
    
    # 1. Initialize Baseline 1: Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_val, y_val)
    
    # 2. Initialize Baseline 2: Decision Tree
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42) # depth limited to avoid heavy overfitting
    dt_model.fit(X_train, y_train)
    dt_metrics = evaluate_model(dt_model, X_val, y_val)
    
    # Print Results Side-by-Side
    print("\n=================== BASELINE MODEL COMPARISON ===================")
    metrics_df = pd.DataFrame([lr_metrics, dt_metrics], index=["Logistic Regression", "Decision Tree"])
    print(metrics_df.round(4))
    print("=================================================================\n")

if __name__ == "__main__":
    main()