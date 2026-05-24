# Churn Prediction — Final Report

## Abstract
This report documents the methodology, experiments, results, and deployment architecture for the Telco churn prediction project. It covers dataset preparation, feature engineering, model training and evaluation, the inference service and UI, monitoring, containerization, and deployment to Azure Container Apps.

## Repository & Files
- Code and API: [app/main.py](app/main.py)
- Container image definition: [Dockerfile](Dockerfile)
- Dependencies: [requirements.txt](requirements.txt)
- Models: `models/` (includes `xgboost.pkl`, `scaler.pkl`, `label_encoders.pkl`, `feature_columns.pkl`)

## 1. Introduction
State the project goals: predict customer churn probability and provide a lightweight, user-friendly UI for business users and stakeholders. Provide brief description of dataset origin (telco/customer usage) and performance targets (e.g., maximize AUC and recall for churn class while keeping latency < 200ms).

## 2. Data & Preprocessing
- Describe raw fields (categorical, numeric, dates). List the features used in the final model (these are saved as `feature_columns.pkl` in `models/`).
- Preprocessing steps used in training:
  - Missing-value handling strategy
  - Categorical encoding (label encoders saved at `models/label_encoders.pkl`)
  - Numeric scaling (scaler in `models/scaler.pkl`)
  - Any feature creation (e.g., `feedback_length`, `sentiment`)

### Reproduction commands (training environment)
Provide commands used to run experiments (example placeholder):

```bash
# create venv and install
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

# run training script (replace with your training script)
python train.py --config configs/train.yaml
```

## 3. Models & Methodology
- Models evaluated: XGBoost (production), Random Forest, Decision Tree, Logistic Regression, Neural Network variants (listed in `models/`).
- Training strategy: stratified train/validation split (typical 80/20), hyperparameter search with grid/random search or Optuna, early stopping for boosted trees.
- Evaluation metrics: ROC AUC, PR AUC, Accuracy, Precision, Recall, F1, Calibration (Brier score), Prediction latency, and model size.

### Hyperparameters (example for XGBoost)
- objective: `binary:logistic`
- n_estimators: 100-1000 (early stop)
- max_depth: 3-10
- learning_rate: 0.01-0.3
- subsample: 0.6-1.0

## 4. Experiments and Results
Include tables and figures here. Suggested content and placeholders:

- Dataset size: N_train, N_val, N_test
- Class distribution: churn vs. non-churn

Performance table (example format):

| Model | ROC AUC | PR AUC | Precision | Recall | F1 | Inference latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost (prod) | 0.92 | 0.68 | 0.72 | 0.64 | 0.68 | 12 |
| Random Forest | 0.89 | 0.63 | 0.70 | 0.60 | 0.65 | 48 |
| LogisticReg | 0.84 | 0.52 | 0.60 | 0.55 | 0.57 | 3 |

Replace the example numbers with your measured results. Include a confusion matrix and ROC curve images (save under `docs/images/`).

### Calibration & Business Impact
- Present calibration plot and thresholds for action (e.g., threshold 0.6 → targeted retention campaign).
- Estimate business KPIs: expected reduction in churn, ROI per campaign, etc. (provide calculations if available).

## 5. Inference Service & UI
- The FastAPI application serves a polished UI at `/` and exposes the prediction API at `/predict` (see [app/main.py](app/main.py)).
- Monitoring endpoints: `/metrics` (Prometheus), `/stats` (aggregated prediction stats), `/system` (process-level CPU/memory).
- Request metrics: `churn_api_requests_total`, `churn_api_request_duration_seconds`, `churn_api_in_flight_requests`, `churn_predictions_total`.
- Middleware adds `X-Process-Time` header used by the UI.

## 6. Containerization
- Dockerfile at the repository root packages the app and model artifacts. Key points:
  - Copies `app/` and `models/` into the image
  - Exposes port `8000`
  - Runs `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}`

Build & run locally:

```bash
docker build -t <acr_name>.azurecr.io/churn-analysis:latest .
# or local test
docker build -t churn-local:latest .
docker run --rm -p 8000:8000 churn-local:latest
```

## 7. Deployment Architecture (Azure Container Apps)
- Recommended configuration for Azure Container Apps:
  - Image: `<your-acr-name>.azurecr.io/churn-analysis:latest`
  - Ingress: External enabled
  - Target port: `8000`
  - Health probe path: `/health`
  - Registry auth: grant `AcrPull` role to the Container App managed identity or provide registry credentials
  - Resource sizing: start with 0.5–1 vCPU and 1–2 GB RAM, add scale rules based on HTTP concurrent requests or CPU

Live app URL:

```text
https://churnanalysis.whitedesert-71f6c6f8.uaenorth.azurecontainerapps.io/
```

Health check URL:

```text
https://churnanalysis.whitedesert-71f6c6f8.uaenorth.azurecontainerapps.io/health
```

Include a diagram (save at `docs/images/deployment-architecture.png`) showing:
- Users → Azure Container App (ingress) → Container (uvicorn) → model artifacts
- Monitoring: Prometheus scrape via managed agent or external exporter; logs to Azure Monitor
- Optional: put behind Application Gateway for WAF, TLS termination, and custom domain

## 8. Monitoring & Observability
- Expose `/metrics` for Prometheus scraping.
- Collect these key metrics: requests/sec, p50/p95 latency, error rate, memory, CPU, prediction distribution (positive rate).
- Alerting suggestions: high error rate (>1%), increased p95 latency (>500ms), sudden drift in positive_rate.

## 9. Security & Secrets
- Store ACR credentials or use managed identity. Do NOT embed credentials in images or repo.
- Validate input on `/predict` and return clear 422 errors for missing/unknown values (already implemented in API).
- TLS termination: enable HTTPS for the Container App and use a custom domain with managed certificate if needed.

## 10. Reproducibility & CI/CD
- Repro steps to create and push image to ACR:

```bash
# log in to Azure
az acr login --name <your-acr-name>
# build and push
docker build -t <your-acr-name>.azurecr.io/churn-analysis:latest .
docker push <your-acr-name>.azurecr.io/churn-analysis:latest
```

- Example GitHub Actions job to build PDF on push and publish artifact (optional):

```yaml
name: Build Report
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install pandoc + TeX
        run: sudo apt-get update && sudo apt-get install -y pandoc texlive-xetex
      - name: Render PDF
        run: pandoc docs/final_report.md -o docs/final_report.pdf --pdf-engine=xelatex
      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: final-report-pdf
          path: docs/final_report.pdf
```

## 11. Commands to generate PDF locally
- Using Pandoc (recommended for high-quality PDF):

```powershell
# Windows PowerShell
choco install pandoc   # or visit https://pandoc.org/installing.html
# install a TeX engine such as TinyTeX or MiKTeX for PDF rendering
pandoc docs/final_report.md -o docs/final_report.pdf --pdf-engine=xelatex
```

- Quick alternative: open `docs/final_report.md` in VS Code and use Print to PDF or use Markdown-to-HTML then print.

## 12. Appendix
- Sample `curl` tests:

```bash
curl -sS http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d @sample_payload.json
```

- Sample payload (save as `sample_payload.json`):

```json
{
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
    "TotalCharges": 844.2,
    "feedback_length": 120,
    "sentiment": -0.25
  }
}
```

---

### Notes for the author
1. Replace placeholder experiment numbers and tables with your measured metrics.
2. Add plots under `docs/images/` and reference them in the Results section.
3. If you want, I can convert this Markdown to PDF for you — tell me whether you want me to run pandoc in this environment or provide a GitHub Actions workflow and I'll create it.


