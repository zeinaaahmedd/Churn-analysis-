import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf

def load_test_split_data():
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
    X_tab_test_full = np.hstack((X_tab.iloc[idx_test].drop(columns=numeric_cols).values, scaler.transform(X_tab.iloc[idx_test][numeric_cols])))
    X_tab_train_full = np.hstack((X_tab.iloc[idx_train].drop(columns=numeric_cols).values, X_tab_train))
    
    X_text_test = df['CustomerFeedback'].values[idx_test]
    y_test = y[idx_test]
    
    return X_tab_train_full, X_tab_test_full, X_text_test, y[idx_train], y_test

def main():
    X_train_tab, X_test_tab, X_text_test, y_train, y_test = load_test_split_data()
    results = {}

    # 1. Baseline Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_tab, y_train)
    results["Logistic Regression"] = [
        accuracy_score(y_test, lr.predict(X_test_tab)),
        precision_score(y_test, lr.predict(X_test_tab)),
        recall_score(y_test, lr.predict(X_test_tab)),
        f1_score(y_test, lr.predict(X_test_tab)),
        roc_auc_score(y_test, lr.predict_proba(X_test_tab)[:, 1])
    ]

    # 2. Baseline Decision Tree
    dt = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train_tab, y_train)
    results["Decision Tree"] = [
        accuracy_score(y_test, dt.predict(X_test_tab)),
        precision_score(y_test, dt.predict(X_test_tab)),
        recall_score(y_test, dt.predict(X_test_tab)),
        f1_score(y_test, dt.predict(X_test_tab)),
        roc_auc_score(y_test, dt.predict_proba(X_test_tab)[:, 1])
    ]

    # 3. Multimodal Neural Network
    nn_model = tf.keras.models.load_model("models/multimodal_nn_model.keras")
    probs_nn = nn_model.predict({"tabular_input": X_test_tab, "text_input": X_text_test}).flatten()
    preds_nn = (probs_nn > 0.5).astype(int)
    results["Neural Network"] = [
        accuracy_score(y_test, preds_nn),
        precision_score(y_test, preds_nn),
        recall_score(y_test, preds_nn),
        f1_score(y_test, preds_nn),
        roc_auc_score(y_test, probs_nn)
    ]

    metrics_df = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]).T
    print("\n=================== UNBIASED HELD-OUT TEST SPLIT RESULTS ===================")
    print(metrics_df.round(4))
    print("============================================================================\n")
    
    metrics_df.to_csv("plots/final_test_metrics.csv")

if __name__ == "__main__":
    main()