import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf

def load_and_prepare_evaluation_data():
    raw_data_path = "data/raw/telco_prep.csv"
    df = pd.read_csv(raw_data_path)
    
    df['CustomerFeedback'] = df['CustomerFeedback'].fillna('').astype(str)
    y = df['Churn'].values
    
    cols_to_drop = ['Unnamed: 0', 'customerID', 'PromptInput', 'CustomerFeedback', 'Churn', 'feedback_length', 'sentiment']
    X_tab = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    numeric_cols = X_tab.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_tab.select_dtypes(include=['object']).columns.tolist()
    
    le = LabelEncoder()
    for col in categorical_cols:
        X_tab[col] = le.fit_transform(X_tab[col].astype(str))
        
    idx_train, idx_temp = train_test_split(np.arange(len(df)), test_size=0.30, random_state=42, stratify=y)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=42, stratify=y[idx_temp])
    
    scaler = StandardScaler()
    X_tab_train = scaler.fit_transform(X_tab.iloc[idx_train][numeric_cols])
    X_tab_val = scaler.transform(X_tab.iloc[idx_val][numeric_cols])
    
    X_tab_train_full = np.hstack((X_tab.iloc[idx_train].drop(columns=numeric_cols).values, X_tab_train))
    X_tab_val_full = np.hstack((X_tab.iloc[idx_val].drop(columns=numeric_cols).values, X_tab_val))
    
    X_text_val = df['CustomerFeedback'].values[idx_val]
    y_val = y[idx_val]
    
    return X_tab_train_full, X_tab_val_full, X_text_val, y[idx_train], y_val

def gather_all_metrics():
    X_train_tab, X_val_tab, X_text_val, y_train, y_val = load_and_prepare_evaluation_data()
    results = {}

    # --- Model 1: Logistic Regression ---
    print("Evaluating Logistic Regression Baseline...")
    lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_tab, y_train)
    results["Logistic Regression"] = [
        accuracy_score(y_val, lr.predict(X_val_tab)),
        precision_score(y_val, lr.predict(X_val_tab)),
        recall_score(y_val, lr.predict(X_val_tab)),
        f1_score(y_val, lr.predict(X_val_tab)),
        roc_auc_score(y_val, lr.predict_proba(X_val_tab)[:, 1])
    ]

    # --- Model 2: Decision Tree ---
    print("Evaluating Decision Tree Baseline...")
    dt = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train_tab, y_train)
    results["Decision Tree"] = [
        accuracy_score(y_val, dt.predict(X_val_tab)),
        precision_score(y_val, dt.predict(X_val_tab)),
        recall_score(y_val, dt.predict(X_val_tab)),
        f1_score(y_val, dt.predict(X_val_tab)),
        roc_auc_score(y_val, dt.predict_proba(X_val_tab)[:, 1])
    ]

    # --- Model 3: Multimodal Neural Network ---
    print("Evaluating Multimodal Neural Network...")
    nn_model = tf.keras.models.load_model("models/multimodal_nn_model.keras")
    
    probs_nn = nn_model.predict({"tabular_input": X_val_tab, "text_input": X_text_val}).flatten()
    preds_nn = (probs_nn > 0.5).astype(int)
    
    results["Neural Network"] = [
        accuracy_score(y_val, preds_nn),
        precision_score(y_val, preds_nn),
        recall_score(y_val, preds_nn),
        f1_score(y_val, preds_nn),
        roc_auc_score(y_val, probs_nn)
    ]

    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    return pd.DataFrame(results, index=metrics_names).T

def plot_visualizations(metrics_df):
    os.makedirs("plots", exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Sentiment vs Churn Distribution
    df_raw = pd.read_csv("data/raw/telco_prep.csv")
    plt.figure(figsize=(9, 5))
    sns.kdeplot(data=df_raw, x="sentiment", hue="Churn", fill=True, common_norm=False, palette="crest", alpha=0.6)
    plt.title("Data Insight: Sentiment Score Distribution by Churn Status")
    plt.xlabel("Customer Feedback Sentiment Score")
    plt.ylabel("Density")
    plt.savefig("plots/data_sentiment_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved plot: plots/data_sentiment_distribution.png")

    # Plot 2: Model Comparison Bar Chart
    plt.figure(figsize=(11, 6))
    metrics_df_melted = metrics_df.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
    metrics_df_melted.rename(columns={"index": "Model"}, inplace=True)
    
    sns.barplot(data=metrics_df_melted, x="Metric", y="Score", hue="Model", palette="Set2")
    plt.ylim(0.4, 1.0)
    plt.title("Model Performance Comparison Matrix (Validation Set)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig("plots/model_comparison_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved plot: plots/model_comparison_matrix.png")

def main():
    metrics_df = gather_all_metrics()
    print("\n=================== FINAL ALL-MODEL METRIC MATRIX ===================")
    print(metrics_df.round(4))
    print("=====================================================================\n")
    plot_visualizations(metrics_df)

if __name__ == "__main__":
    main()