import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_and_prepare_nn_data():
    raw_data_path = "data/raw/telco_prep.csv"
    df = pd.read_csv(raw_data_path)
    
    # Clean text input
    df['CustomerFeedback'] = df['CustomerFeedback'].fillna('').astype(str)
    y = df['Churn'].values
    
    # Process Tabular parts
    cols_to_drop = ['Unnamed: 0', 'customerID', 'PromptInput', 'CustomerFeedback', 'Churn', 'feedback_length', 'sentiment']
    X_tab = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    numeric_cols = X_tab.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_tab.select_dtypes(include=['object']).columns.tolist()
    
    le = LabelEncoder()
    for col in categorical_cols:
        X_tab[col] = le.fit_transform(X_tab[col].astype(str))
        
    X_text = df['CustomerFeedback'].values
    
    # Split indices uniformly to keep tabular and text aligned
    idx_train, idx_temp = train_test_split(np.arange(len(df)), test_size=0.30, random_state=42, stratify=y)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=42, stratify=y[idx_temp])
    
    # Scale Tabular features
    scaler = StandardScaler()
    X_tab_train = scaler.fit_transform(X_tab.iloc[idx_train][numeric_cols])
    X_tab_val = scaler.transform(X_tab.iloc[idx_val][numeric_cols])
    X_tab_test = scaler.transform(X_tab.iloc[idx_test][numeric_cols])
    
    # Recombine scaled numeric back with encoded categorical features
    X_tab_train_full = np.hstack((X_tab.iloc[idx_train].drop(columns=numeric_cols).values, X_tab_train))
    X_tab_val_full = np.hstack((X_tab.iloc[idx_val].drop(columns=numeric_cols).values, X_tab_val))
    X_tab_test_full = np.hstack((X_tab.iloc[idx_test].drop(columns=numeric_cols).values, X_tab_test))
    
    return (X_tab_train_full, X_text[idx_train]), (X_tab_val_full, X_text[idx_val]), (X_tab_test_full, X_text[idx_test]), y[idx_train], y[idx_val], y[idx_test]

def build_multimodal_nn(tabular_shape, max_tokens=5000, output_sequence_length=100):
    # 1. Branch A: Tabular Input
    tabular_input = layers.Input(shape=(tabular_shape,), name="tabular_input")
    tab_dense = layers.Dense(32, activation="relu")(tabular_input)
    tab_dense = layers.BatchNormalization()(tab_dense)
    tab_dense = layers.Dropout(0.3)(tab_dense)
    
    # 2. Branch B: Raw Text Input
    text_input = layers.Input(shape=(1,), dtype=tf.string, name="text_input")
    
    # Vectorize strings to integers tokens
    vectorize_layer = layers.TextVectorization(max_tokens=max_tokens, output_sequence_length=output_sequence_length)
    # Note: Simplification for script initialization: fit vectorizer internally or mock
    
    text_vect = vectorize_layer(text_input)
    text_emb = layers.Embedding(input_dim=max_tokens, output_dim=16, input_length=output_sequence_length)(text_vect)
    text_flat = layers.GlobalAveragePooling1D()(text_emb)
    text_dense = layers.Dense(16, activation="relu")(text_flat)
    
    # 3. Fusion: Concatenate both feature representations
    fused = layers.Concatenate()([tab_dense, text_dense])
    
    # 4. Final Classification Layers
    fc = layers.Dense(16, activation="relu")(fused)
    fc = layers.Dropout(0.2)(fc)
    output = layers.Dense(1, activation="sigmoid", name="output")(fc)
    
    model = models.Model(inputs=[tabular_input, text_input], outputs=output)
    
    # Adapt vectorizer mapping logic placeholder (will handle during raw training runtime)
    return model, vectorize_layer

def main():
    train_data, val_data, test_data, y_train, y_val, y_test = load_and_prepare_nn_data()
    X_tab_train, X_text_train = train_data
    X_tab_val, X_text_val = val_data
    
    model, vectorize_layer = build_multimodal_nn(X_tab_train.shape[1])
    
    # Adapt text vectorizer explicitly to our training corpus text
    vectorize_layer.adapt(X_text_train)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    print("\nTraining Multimodal Neural Network...")
    # Add class weighting because of data skew (73% non-churn, 27% churn)
    class_weights = {0: 1.0, 1: 2.7}
    
    model.fit(
        x={"tabular_input": X_tab_train, "text_input": X_text_train},
        y=y_train,
        validation_data=({"tabular_input": X_tab_val, "text_input": X_text_val}, y_val),
        epochs=15,
        batch_size=32,
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate
    probs = model.predict({"tabular_input": X_tab_val, "text_input": X_text_val}).flatten()
    preds = (probs > 0.5).astype(int)
    
    print("\n=================== NEURAL NETWORK PERFORMANCE (VAL) =================== ")
    print(f"Accuracy : {accuracy_score(y_val, preds):.4f}")
    print(f"Precision: {precision_score(y_val, preds):.4f}")
    print(f"Recall   : {recall_score(y_val, preds):.4f}")
    print(f"F1-Score : {f1_score(y_val, preds):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_val, probs):.4f}")
    print("=========================================================================\n")
    
    # Save the deep model configuration
    os.makedirs("models", exist_ok=True)
    model.save("models/multimodal_nn_model.keras")
    print("Neural network saved successfully to models/multimodal_nn_model.keras")

if __name__ == "__main__":
    main()