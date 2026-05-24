import os
from pyexpat import model
import pandas as pd
import numpy as np
import json
import yaml
import dagshub
import joblib
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

dagshub.init(
    repo_owner='MennaSherieff',
    repo_name='Churn-analysis-',
    mlflow=True
)

mlflow.set_experiment("Classical_Model_Experiments")
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

train_params = params["train"]

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
    
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=train_params["logistic_regression"]["max_iter"],
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=train_params["decision_tree"]["max_depth"],
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=train_params["random_forest"]["n_estimators"],
            max_depth=train_params["random_forest"]["max_depth"],
            random_state=train_params["random_forest"]["random_state"]
        ),

        "XGBoost": XGBClassifier(
            n_estimators=train_params["xgboost"]["n_estimators"],
            learning_rate=train_params["xgboost"]["learning_rate"],
            max_depth=train_params["xgboost"]["max_depth"],
            random_state=train_params["xgboost"]["random_state"],
            eval_metric="logloss"
        )
    }
    all_metrics = {}

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        with mlflow.start_run(run_name=model_name):

            model.fit(X_train, y_train)

            metrics = evaluate_model(model, X_val, y_val)

            all_metrics[model_name] = metrics

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            
            model_path = f"models/{model_name.replace(' ', '_').lower()}.pkl"
            joblib.dump(model, model_path)

            mlflow.log_artifact(model_path)
    
    metrics_df = pd.DataFrame(all_metrics).T

    print(metrics_df.round(4))

    metrics_df.to_json("metrics_classical.json", indent=4)

    mlflow.log_artifact("metrics_classical.json")

if __name__ == "__main__":
    main()