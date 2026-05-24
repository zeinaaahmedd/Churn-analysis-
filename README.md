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
├── app/
│   └── main.py                  # FastAPI inference service
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── xgboost.pkl
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── multimodal_nn_model.keras
│   ├── label_encoders.pkl       # categorical encoders for inference
│   ├── scaler.pkl               # feature scaler for inference
│   └── feature_columns.pkl     # expected column order for inference
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
├── requirements.txt
├── Dockerfile
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
* FastAPI
* Uvicorn
* Docker

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

* [x] Add FastAPI deployment layer
* [x] Dockerize inference service
* [x] Deploy model as API service
* [ ] Add CI/CD pipeline for training automation

---

# API

The inference service is built with FastAPI and serves the best-performing XGBoost model.

## Web Interface

Open the root page at `/` to use the browser interface. It renders the customer input form directly from the saved feature encoders and posts the payload to `/predict`.

## Running locally

```bash
uvicorn app.main:app --reload --port 8000
```

Interactive docs are available at `http://localhost:8000/docs`.

## Deploying on Azure

This repository is ready for Azure Kubernetes Service (AKS).

### 1. Build the Docker image

```bash
docker build -t churn-analysis .
```

### 2. Create an Azure Container Registry and push the image

```bash
az login
az group create --name churn-rg --location eastus
az acr create --resource-group churn-rg --name <your-acr-name> --sku Basic
az acr login --name <your-acr-name>
docker tag churn-analysis <your-acr-name>.azurecr.io/churn-analysis:latest
docker push <your-acr-name>.azurecr.io/churn-analysis:latest
```

### 3. Create AKS and connect kubectl

```bash
az aks create --resource-group churn-rg --name churn-aks --node-count 2 --enable-managed-identity --attach-acr <your-acr-name>
az aks get-credentials --resource-group churn-rg --name churn-aks
```

### 4. Deploy the service

Apply the Azure AKS manifests in [azure/aks/deployment.yaml](azure/aks/deployment.yaml) and [azure/aks/service.yaml](azure/aks/service.yaml).

Before applying, replace `<your-acr-name>` in [azure/aks/deployment.yaml](azure/aks/deployment.yaml) with your real Azure Container Registry name.

```bash
kubectl apply -f azure/aks/deployment.yaml
kubectl apply -f azure/aks/service.yaml
kubectl get svc churn-analysis
```

### 5. Open the public endpoint

When the `EXTERNAL-IP` appears, open it in the browser. The root route `/` is the UI, and the service also exposes `/health`, `/predict`, `/metrics`, `/stats`, and `/system`.

### Optional local Docker test

```bash
docker run -p 8000:8000 churn-analysis
```

### Notes

The app exposes `/health`, `/predict`, `/metrics`, `/stats`, and `/system`.

The key endpoints are:

* `/` - browser UI
* `/health` - health check for the service
* `/predict` - JSON prediction API
* `/metrics` - Prometheus-style runtime metrics for latency, throughput, and error rate
* `/stats` - simple model-level summary of prediction volume and class balance
* `/system` - CPU and memory usage for the running process

## GET /health

Returns service liveness.

```bash
curl http://localhost:8000/health
```

Response:

```json
{"status": "ok"}
```

## POST /predict

Accepts a raw customer record and returns a churn probability with a binary prediction.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "gender": "female",
      "SeniorCitizen": 0,
      "Partner": "yes",
      "Dependents": "no",
      "tenure": 12,
      "PhoneService": "yes",
      "MultipleLines": "no",
      "InternetService": "fiber optic",
      "OnlineSecurity": "no",
      "OnlineBackup": "no",
      "DeviceProtection": "no",
      "TechSupport": "no",
      "StreamingTV": "yes",
      "StreamingMovies": "yes",
      "Contract": "month-to-month",
      "PaperlessBilling": "yes",
      "PaymentMethod": "electronic check",
      "MonthlyCharges": 70.35,
      "TotalCharges": 844.20,
      "feedback_length": 120,
      "sentiment": -0.25
    }
  }'
```

Response:

```json
{"churn_probability": 0.8213, "prediction": 1}
```

**Error responses:**
- `422` — a required field is missing, or a categorical field contains a value not seen during training.

## Monitoring

The service exposes three lightweight operational views:

* `/metrics` returns Prometheus-format application metrics, including request count, request latency, and in-flight requests.
* `/stats` returns a JSON summary of cumulative churn predictions, positive/negative counts, positive rate, and average predicted probability.
* `/system` returns current CPU and memory usage for the application process.

Example calls:

```bash
curl http://localhost:8000/metrics
curl http://localhost:8000/stats
curl http://localhost:8000/system
```

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
