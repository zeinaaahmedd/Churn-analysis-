import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import dagshub
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


dagshub.init(
    repo_owner='MennaSherieff',
    repo_name='Churn-analysis-',
    mlflow=True
)

mlflow.set_experiment("Full_Model_Comparison")


def load_and_prepare_evaluation_data():
    """Load preprocessed data from prepare.py"""
    processed_dir = "data/processed"
    
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_val = pd.read_csv(os.path.join(processed_dir, "X_val.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).values.ravel()
    y_val = pd.read_csv(os.path.join(processed_dir, "y_val.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).values.ravel()
    
    # Convert to numpy arrays
    X_train = X_train.values.astype(np.float32)
    X_val = X_val.values.astype(np.float32)
    
    # Note: Your neural network doesn't use text, so we return None for text
    return X_train, X_val, None, y_train, y_val


def save_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.title(f"{model_name} Confusion Matrix")

    path = f"plots/confusion_matrix_{model_name.lower().replace(' ', '_')}.png"

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    return path


def plot_roc_curves(roc_data):
    plt.figure(figsize=(10, 7))

    for name, (fpr, tpr, auc_score) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()

    path = "plots/roc_curve_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    return path


def plot_precision_recall_curves(pr_data):
    plt.figure(figsize=(10, 7))

    for name, (precision, recall) in pr_data.items():
        plt.plot(recall, precision, label=name)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.legend()

    path = "plots/pr_curve_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    return path


def gather_all_metrics():
    X_train_tab, X_val_tab, X_text_val, y_train, y_val = load_and_prepare_evaluation_data()

    results = {}
    roc_data = {}
    pr_data = {}

    mlflow.start_run(run_name="Full_Evaluation_Run")

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000).fit(X_train_tab, y_train)
    lr_probs = lr.predict_proba(X_val_tab)[:, 1]
    lr_preds = lr.predict(X_val_tab)

    results["Logistic Regression"] = [
        accuracy_score(y_val, lr_preds),
        precision_score(y_val, lr_preds),
        recall_score(y_val, lr_preds),
        f1_score(y_val, lr_preds),
        roc_auc_score(y_val, lr_probs)
    ]

    fpr, tpr, _ = roc_curve(y_val, lr_probs)
    roc_data["Logistic Regression"] = (fpr, tpr, roc_auc_score(y_val, lr_probs))
    pr_data["Logistic Regression"] = precision_recall_curve(y_val, lr_probs)[:2]

    save_confusion_matrix(y_val, lr_preds, "Logistic Regression")

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=5).fit(X_train_tab, y_train)
    dt_probs = dt.predict_proba(X_val_tab)[:, 1]
    dt_preds = dt.predict(X_val_tab)

    results["Decision Tree"] = [
        accuracy_score(y_val, dt_preds),
        precision_score(y_val, dt_preds),
        recall_score(y_val, dt_preds),
        f1_score(y_val, dt_preds),
        roc_auc_score(y_val, dt_probs)
    ]


    fpr, tpr, _ = roc_curve(y_val, dt_probs)
    roc_data["Decision Tree"] = (fpr, tpr, roc_auc_score(y_val, dt_probs))
    pr_data["Decision Tree"] = precision_recall_curve(y_val, dt_probs)[:2]

    save_confusion_matrix(y_val, dt_preds, "Decision Tree")

    # Neural Network
    nn = tf.keras.models.load_model("models/multimodal_nn_model.keras")

    nn_probs = nn.predict(X_val_tab).flatten() 

    nn_preds = (nn_probs > 0.5).astype(int)

    results["Neural Network"] = [
        accuracy_score(y_val, nn_preds),
        precision_score(y_val, nn_preds),
        recall_score(y_val, nn_preds),
        f1_score(y_val, nn_preds),
        roc_auc_score(y_val, nn_probs)
    ]

    fpr, tpr, _ = roc_curve(y_val, nn_probs)
    roc_data["Neural Network"] = (fpr, tpr, roc_auc_score(y_val, nn_probs))
    pr_data["Neural Network"] = precision_recall_curve(y_val, nn_probs)[:2]

    save_confusion_matrix(y_val, nn_preds, "Neural Network")

    metrics_df = pd.DataFrame(results).T

    mlflow.log_metrics({
        "lr_f1": results["Logistic Regression"][3],
        "dt_f1": results["Decision Tree"][3],
        "nn_f1": results["Neural Network"][3]
    })

    return metrics_df, roc_data, pr_data


def plot_visualizations(metrics_df):
    os.makedirs("plots", exist_ok=True)
    sns.set_theme(style="whitegrid")

    df_raw = pd.read_csv("data/raw/telco_prep.csv")

    plt.figure(figsize=(9, 5))
    sns.kdeplot(data=df_raw, x="sentiment", hue="Churn", fill=True, alpha=0.6)

    plt.title("Sentiment vs Churn")
    plt.savefig("plots/data_sentiment_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 6))

    melted = metrics_df.reset_index().melt(id_vars="index")
    melted.columns = ["Model", "Metric", "Score"]

    sns.barplot(data=melted, x="Metric", y="Score", hue="Model")

    plt.ylim(0.4, 1.0)
    plt.title("Model Comparison")
    plt.savefig("plots/model_comparison_matrix.png")
    plt.close()


def main():
    metrics_df, roc_data, pr_data = gather_all_metrics()

    print(metrics_df)

    plot_visualizations(metrics_df)

    plot_roc_curves(roc_data)
    plot_precision_recall_curves(pr_data)

    mlflow.log_artifacts("plots")

    mlflow.end_run()


if __name__ == "__main__":
    main()