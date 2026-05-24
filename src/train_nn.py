import os
import json
import yaml
import dagshub
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

dagshub.init(repo_owner="MennaSherieff", repo_name="Churn-analysis-", mlflow=True)

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

nn_params = params["train_nn"]

LEARNING_RATE = nn_params["learning_rate"]
EPOCHS = nn_params["epochs"]
BATCH_SIZE = nn_params["batch_size"]

TAB_DENSE = nn_params["architecture"]["tabular_dense"]
FUSION_DENSE = nn_params["architecture"]["fusion_dense"]

DROP_TAB = nn_params["dropout"]["tabular"]
DROP_FUS = nn_params["dropout"]["fusion"]


def load_processed_data():
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
    X_test = X_test.values.astype(np.float32)
    
    print(f"Loaded data shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(input_dim):
    inp = layers.Input(shape=(input_dim,))
    
    x = layers.Dense(TAB_DENSE, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROP_TAB)(x)
    
    x = layers.Dense(FUSION_DENSE, activation="relu")(x)
    x = layers.Dropout(DROP_FUS)(x)
    
    out = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(inputs=inp, outputs=out)
    
    return model


def evaluate(model, X, y):
    probs = model.predict(X).ravel()
    preds = (probs > 0.5).astype(int)
    
    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1_score": f1_score(y, preds),
        "roc_auc": roc_auc_score(y, probs)
    }


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    
    with mlflow.start_run():
        
        model = build_model(X_train.shape[1])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        early = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
        
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early],
            verbose=1
        )
        
        metrics = evaluate(model, X_test, y_test)
        
        print("\nNeural Network Test Metrics:")
        print(metrics)
        
        mlflow.log_metrics(metrics)
        
        os.makedirs("models", exist_ok=True)
        model_path = "models/multimodal_nn_model.keras"
        model.save(model_path)
        
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        mlflow.log_artifact(model_path)
        mlflow.log_artifact("metrics.json")


if __name__ == "__main__":
    main()