import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import tensorflow as tf
import yaml

# Load parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

train_params = params["train"]

def load_processed_test_data():
    """Load the preprocessed test data from prepare.py"""
    processed_dir = "data/processed"
    
    # Load the test sets that were saved by prepare.py
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).values.ravel()
    
    # Load training data for the baseline models
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).values.ravel()
    
    # Convert to numpy arrays
    X_train = X_train.values.astype(np.float32)
    X_test = X_test.values.astype(np.float32)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Training samples: {len(y_train)}, Test samples: {len(y_test)}")
    
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = load_processed_test_data()
    results = {}

    print("EVALUATING ON HELD-OUT TEST SET")

    # 1. Baseline Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(
        max_iter=train_params["logistic_regression"]["max_iter"],
        random_state=42
    )
    lr.fit(X_train, y_train)
    
    lr_preds = lr.predict(X_test)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    
    results["Logistic Regression"] = [
        accuracy_score(y_test, lr_preds),
        precision_score(y_test, lr_preds),
        recall_score(y_test, lr_preds),
        f1_score(y_test, lr_preds),
        roc_auc_score(y_test, lr_probs)
    ]
    print(f"Logistic Regression - Accuracy: {results['Logistic Regression'][0]:.4f}")

    # 2. Decision Tree
    print("\nTraining Decision Tree...")
    dt = DecisionTreeClassifier(
        max_depth=train_params["decision_tree"]["max_depth"],
        random_state=42
    )
    dt.fit(X_train, y_train)
    
    dt_preds = dt.predict(X_test)
    dt_probs = dt.predict_proba(X_test)[:, 1]
    
    results["Decision Tree"] = [
        accuracy_score(y_test, dt_preds),
        precision_score(y_test, dt_preds),
        recall_score(y_test, dt_preds),
        f1_score(y_test, dt_preds),
        roc_auc_score(y_test, dt_probs)
    ]
    print(f"Decision Tree - Accuracy: {results['Decision Tree'][0]:.4f}")

    # 3. Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=train_params["random_forest"]["n_estimators"],
        max_depth=train_params["random_forest"]["max_depth"],
        random_state=train_params["random_forest"]["random_state"]
    )
    rf.fit(X_train, y_train)
    
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    
    results["Random Forest"] = [
        accuracy_score(y_test, rf_preds),
        precision_score(y_test, rf_preds),
        recall_score(y_test, rf_preds),
        f1_score(y_test, rf_preds),
        roc_auc_score(y_test, rf_probs)
    ]
    print(f"Random Forest - Accuracy: {results['Random Forest'][0]:.4f}")

    # 4. XGBoost
    print("\nTraining XGBoost...")
    xgb = XGBClassifier(
        n_estimators=train_params["xgboost"]["n_estimators"],
        learning_rate=train_params["xgboost"]["learning_rate"],
        max_depth=train_params["xgboost"]["max_depth"],
        random_state=train_params["xgboost"]["random_state"],
        eval_metric="logloss"
    )
    xgb.fit(X_train, y_train)
    
    xgb_preds = xgb.predict(X_test)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    
    results["XGBoost"] = [
        accuracy_score(y_test, xgb_preds),
        precision_score(y_test, xgb_preds),
        recall_score(y_test, xgb_preds),
        f1_score(y_test, xgb_preds),
        roc_auc_score(y_test, xgb_probs)
    ]
    print(f"XGBoost - Accuracy: {results['XGBoost'][0]:.4f}")

    # 5. Neural Network
    print("\nLoading Neural Network...")
    nn_model = tf.keras.models.load_model("models/multimodal_nn_model.keras")
    
    # Get the expected input shape
    expected_shape = nn_model.input_shape
    print(f"Model expects input shape: {expected_shape}")
    print(f"Actual test data shape: {X_test.shape}")
    
    # Make predictions
    nn_probs = nn_model.predict(X_test).flatten()
    nn_preds = (nn_probs > 0.5).astype(int)
    
    results["Neural Network"] = [
        accuracy_score(y_test, nn_preds),
        precision_score(y_test, nn_preds),
        recall_score(y_test, nn_preds),
        f1_score(y_test, nn_preds),
        roc_auc_score(y_test, nn_probs)
    ]
    print(f"Neural Network - Accuracy: {results['Neural Network'][0]:.4f}")

    # Create results dataframe
    metrics_df = pd.DataFrame(
        results, 
        index=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    ).T
    
    print("FINAL MODEL COMPARISON ON HELD-OUT TEST SET")
    print(metrics_df.round(4))
    
    # Find the best model based on F1-Score
    metrics_df["Score"] = (
        0.5 * metrics_df["F1-Score"] +
        0.3 * metrics_df["ROC-AUC"] +
        0.2 * metrics_df["Recall"]
    )
    best_model = metrics_df["Score"].idxmax()
    print(f"\nBest Model: {best_model} with Score: {metrics_df.loc[best_model, 'Score']:.4f}")
    
    # Save results
    os.makedirs("plots", exist_ok=True)
    metrics_df.to_csv("plots/final_test_metrics.csv")
    print("\nResults saved to: plots/final_test_metrics.csv")

if __name__ == "__main__":
    main()