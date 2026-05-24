# Telco Customer Churn Prediction — End-to-End MLOps Pipeline

This project is a full **Machine Learning Operations (MLOps)** pipeline for predicting customer churn using both classical machine learning models and a deep learning model. It integrates **DVC for pipeline orchestration**, **MLflow + DagsHub for experiment tracking**, and reproducible training/evaluation workflows.

---

# Project Overview

The goal of this project is to build a **production-ready churn prediction system** that:

- Processes raw telecom customer data
- Trains multiple ML models (classical + deep learning)
- Evaluates models on a held-out test set
- Tracks experiments using MLflow (via DagsHub)
- Ensures full reproducibility using DVC pipelines
- Produces visual insights and model comparisons

---

# Models Used

The project benchmarks multiple models:

### Classical Machine Learning
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### Deep Learning
- Fully Connected Neural Network (TensorFlow/Keras)

---

# Pipeline Architecture (DVC)

The entire workflow is orchestrated using **DVC pipelines**:

```bash
dvc repro
````

## Pipeline Stages

### 1. Data Preparation (`prepare`)

* Loads raw dataset
* Cleans and preprocesses data
* Splits into train/validation/test sets
* Outputs:

  * `X_train.csv`, `X_val.csv`, `X_test.csv`
  * `y_train.csv`, `y_val.csv`, `y_test.csv`

---

### 2. Classical Model Training (`train`)

* Trains multiple ML models
* Evaluates on validation set
* Logs metrics to MLflow (via DagsHub)

Tracked models:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost

---

### 3. Neural Network Training (`train_nn`)

* Builds a fully connected neural network
* Uses early stopping for regularization
* Evaluates on test set
* Saves model as:

```
models/multimodal_nn_model.keras
```

* Logs metrics + artifacts to MLflow

---

### 4. Visualization (`visualize`)

Generates:

* Model comparison plots
* Data distribution analysis
* Confusion matrix / evaluation charts

Outputs saved to:

```
plots/
```

---

### 5. Final Evaluation (`test_evaluation`)

* Evaluates all models on held-out test set
* Compares:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * ROC-AUC
* Generates final leaderboard

Output:

```
plots/final_test_metrics.csv
```

---

# Experiment Tracking (MLflow + DagsHub)

All experiments are tracked using **MLflow integrated with DagsHub**.

## 🔗 DagsHub Repository

[https://dagshub.com/MennaSherieff/Churn-analysis-](https://dagshub.com/MennaSherieff/Churn-analysis-)

## MLflow Tracking UI

Accessible via DagsHub:

```
https://dagshub.com/MennaSherieff/Churn-analysis-.mlflow
```

Tracked metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Model parameters

Each model run is logged separately for full reproducibility.

---

# Project Structure

```
Churn-analysis/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── decision_tree.pkl
│   ├── logestic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── multimodal_nn_model.keras
│
├── plots/
│   ├── model_comparison_matrix.png
│   ├── data_sentiment_distribution.png
│   └── final_test_metrics.csv
│
├── src/
│   ├── prepare.py
│   ├── train.py
│   ├── train_nn.py
│   ├── evaluate_and_plot.py
│   └── final_test_evaluation.py
│
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── metrics.json
└── README.md
```

---

# Reproducibility (DVC)

To reproduce the entire pipeline:

```bash
dvc repro
```

To track changes:

```bash
git add dvc.lock
git commit -m "update pipeline"
```

To push data/artifacts (if remote is configured):

```bash
dvc push
```

---

# Model Evaluation Summary

Best model selection is based on a weighted scoring approach:

```
Score = 0.5 * F1 + 0.3 * ROC-AUC + 0.2 * Recall
```

Final evaluation compares all models on a **held-out test set**.

---

# Neural Network Architecture

* Fully connected dense network
* Batch Normalization
* Dropout regularization
* Sigmoid output for binary classification
* Early stopping used to prevent overfitting

---

# Tech Stack

* Python
* Scikit-learn
* XGBoost
* TensorFlow / Keras
* Pandas / NumPy
* Matplotlib / Seaborn
* MLflow
* DagsHub
* DVC

---

# How to Run the Project

## 1. Clone repository

```bash
git clone https://github.com/mennasherieff/Churn-analysis.git
cd Churn-analysis
```

---

## 2. Create environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run full pipeline

```bash
dvc repro
```

---

## 5. Run individual stages (optional)

```bash
python src/prepare.py
python src/train.py
python src/train_nn.py
python src/final_test_evaluation.py
```

---

# Key Results

* Best classical model: Logistic Regression / XGBoost (close performance)
* Neural Network performs competitively but not best
* ROC-AUC across models ~0.94+
* Strong generalization on held-out test set

---

# Outputs

Generated artifacts include:

* Trained models
* MLflow experiment logs
* Evaluation CSVs
* Confusion matrices
* ROC/PR curves
* Model comparison plots

---

# Future Improvements

Planned upgrades:

* [ ] Add FastAPI deployment layer
* [ ] Dockerize full pipeline
* [ ] Deploy model as API service
* [ ] Add CI/CD pipeline for training automation

---

# Author

Built as part of an end-to-end MLOps learning project focusing on:

* production ML pipelines
* reproducibility
* experiment tracking
* model benchmarking

---

# Acknowledgements

* DVC for pipeline orchestration
* DagsHub for MLflow integration
* Scikit-learn & TensorFlow community
