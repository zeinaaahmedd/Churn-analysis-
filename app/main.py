import json
import os
from threading import Lock
from time import perf_counter
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
import psutil

app = FastAPI(title="Churn Prediction API")
PROCESS = psutil.Process(os.getpid())

REQUEST_COUNT = Counter(
    "churn_api_requests_total",
    "Total HTTP requests received by the churn service",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "churn_api_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)
IN_FLIGHT_REQUESTS = Gauge(
    "churn_api_in_flight_requests",
    "Number of HTTP requests currently being processed",
)
PREDICTION_TOTAL = Counter(
    "churn_predictions_total",
    "Total churn predictions generated",
)

STATS_LOCK = Lock()
PREDICTION_COUNTS = {"positive": 0, "negative": 0}
PREDICTION_SUM = 0.0

MODEL = joblib.load("models/xgboost.pkl")
LABEL_ENCODERS = joblib.load("models/label_encoders.pkl")
SCALER = joblib.load("models/scaler.pkl")
FEATURE_COLS = joblib.load("models/feature_columns.pkl")


@app.middleware("http")
async def record_http_metrics(request, call_next):
    start_time = perf_counter()
    IN_FLIGHT_REQUESTS.inc()

    try:
        response = await call_next(request)
    except Exception:
        REQUEST_COUNT.labels(request.method, request.url.path, "500").inc()
        raise
    finally:
        IN_FLIGHT_REQUESTS.dec()

    duration = perf_counter() - start_time
    REQUEST_LATENCY.labels(request.method, request.url.path).observe(duration)
    REQUEST_COUNT.labels(request.method, request.url.path, str(response.status_code)).inc()
    response.headers["X-Process-Time"] = f"{duration:.4f}"
    return response

NUMERIC_COLS = [column for column in FEATURE_COLS if column not in LABEL_ENCODERS]

INTEGER_HINTS = {"SeniorCitizen", "tenure", "feedback_length"}
NUMBER_DEFAULTS = {
        "SeniorCitizen": 0,
        "tenure": 0,
        "MonthlyCharges": 0.0,
        "TotalCharges": 0.0,
        "feedback_length": 0,
        "sentiment": 0.0,
}


class CustomerRecord(BaseModel):
        features: Dict[str, Any]


def prettify_field_name(field_name: str) -> str:
        return field_name.replace("_", " ").title()


def build_field_config() -> list[dict[str, Any]]:
        field_config: list[dict[str, Any]] = []

        for field_name in FEATURE_COLS:
                if field_name in LABEL_ENCODERS:
                        options = [str(option) for option in LABEL_ENCODERS[field_name].classes_.tolist()]
                        field_config.append(
                                {
                                        "name": field_name,
                                        "label": prettify_field_name(field_name),
                                        "kind": "select",
                                        "options": options,
                                        "default": options[0] if options else "",
                                }
                        )
                        continue

                field_config.append(
                        {
                                "name": field_name,
                                "label": prettify_field_name(field_name),
                                "kind": "number",
                                "step": "1" if field_name in INTEGER_HINTS else "any",
                                "default": NUMBER_DEFAULTS.get(field_name, 0),
                        }
                )

        return field_config


FIELD_CONFIG = build_field_config()

FIELD_SECTIONS = [
    {
        "title": "Account basics",
        "description": "Core customer profile and contract details.",
        "fields": {
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        },
    },
    {
        "title": "Services used",
        "description": "Connection and add-on services that often affect churn.",
        "fields": {
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        },
    },
    {
        "title": "Billing and engagement",
        "description": "Charges and engagement signals used by the model.",
        "fields": {
            "MonthlyCharges",
            "TotalCharges",
            "feedback_length",
            "sentiment",
        },
    },
]

DEMO_FEATURES = {
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
    "sentiment": -0.25,
}


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
        field_config_json = json.dumps(FIELD_CONFIG)
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Churn Prediction Studio</title>
    <style>
        :root {{
            color-scheme: light;
            --bg: #f4efe8;
            --panel: rgba(255, 255, 255, 0.82);
            --panel-strong: #ffffff;
            --text: #132238;
            --muted: #52606d;
            --accent: #0f766e;
            --accent-2: #1d4ed8;
            --border: rgba(19, 34, 56, 0.12);
            --shadow: 0 24px 60px rgba(15, 23, 42, 0.12);
            --danger: #b42318;
            --success: #127f5d;
        }}

        * {{ box-sizing: border-box; }}

        body {{
            margin: 0;
            font-family: Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(29, 78, 216, 0.14), transparent 24%),
                linear-gradient(180deg, #faf7f2 0%, var(--bg) 100%);
            min-height: 100vh;
        }}

        .shell {{
            width: min(1180px, calc(100% - 32px));
            margin: 0 auto;
            padding: 32px 0 40px;
        }}

        .hero {{
            display: grid;
            gap: 18px;
            grid-template-columns: 1fr;
            align-items: start;
            margin-bottom: 22px;
        }}

        .eyebrow {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border: 1px solid var(--border);
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.7);
            color: var(--accent);
            font-size: 0.9rem;
            font-weight: 700;
            width: fit-content;
        }}

        h1 {{
            margin: 14px 0 10px;
            font-size: clamp(2.4rem, 5vw, 4.8rem);
            line-height: 0.95;
            letter-spacing: -0.05em;
            max-width: 11ch;
        }}

        .lede {{
            margin: 0;
            max-width: 60ch;
            color: var(--muted);
            font-size: 1.05rem;
            line-height: 1.7;
        }}

        .grid {{
            display: grid;
            grid-template-columns: 1.25fr 0.75fr;
            gap: 18px;
            align-items: start;
        }}

        .panel {{
            border: 1px solid var(--border);
            border-radius: 28px;
            background: var(--panel);
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
            overflow: hidden;
        }}

        .panel-header {{
            padding: 22px 24px 0;
        }}

        .panel-subhead {{
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            padding: 0 24px 0;
            margin-top: 14px;
        }}

        .step-chip {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 9px 12px;
            border-radius: 999px;
            background: rgba(15, 118, 110, 0.08);
            color: var(--accent);
            border: 1px solid rgba(15, 118, 110, 0.14);
            font-size: 0.9rem;
            font-weight: 700;
        }}

        .section-block {{
            padding: 18px 0 0;
        }}

        .section-block + .section-block {{
            margin-top: 18px;
            border-top: 1px solid rgba(19, 34, 56, 0.08);
        }}

        .section-head {{
            display: grid;
            gap: 6px;
            margin-bottom: 14px;
        }}

        .section-head h3 {{
            margin: 0;
            font-size: 1.02rem;
            letter-spacing: -0.02em;
        }}

        .section-head p {{
            margin: 0;
            font-size: 0.94rem;
            color: var(--muted);
        }}

        .panel h2 {{
            margin: 0 0 8px;
            font-size: 1.4rem;
            letter-spacing: -0.03em;
        }}

        .panel p {{
            margin: 0;
            color: var(--muted);
            line-height: 1.6;
        }}

        form {{ padding: 20px 24px 24px; }}

        .form-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
        }}

        .field {{ display: grid; gap: 8px; }}

        .field.full {{ grid-column: 1 / -1; }}

        label {{
            font-size: 0.9rem;
            font-weight: 700;
            color: #243b53;
        }}

        input, select, button {{ font: inherit; }}

        input, select {{
            width: 100%;
            padding: 12px 14px;
            border-radius: 14px;
            border: 1px solid rgba(19, 34, 56, 0.15);
            background: rgba(255, 255, 255, 0.92);
            color: var(--text);
            transition: border-color 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease;
        }}

        input:focus, select:focus {{
            outline: none;
            border-color: rgba(15, 118, 110, 0.55);
            box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.12);
            transform: translateY(-1px);
        }}

        .actions {{
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 18px;
        }}

        .actions .secondary {{
            background: rgba(15, 118, 110, 0.08);
            color: var(--accent);
            border: 1px solid rgba(15, 118, 110, 0.14);
        }}

        button {{
            border: none;
            border-radius: 999px;
            padding: 14px 18px;
            cursor: pointer;
            font-weight: 700;
        }}

        .primary {{
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            color: white;
            box-shadow: 0 16px 36px rgba(15, 118, 110, 0.24);
        }}

        .ghost {{
            background: rgba(255, 255, 255, 0.8);
            color: var(--text);
            border: 1px solid var(--border);
        }}

        .result {{
            padding: 24px;
            display: grid;
            gap: 16px;
            position: sticky;
            top: 18px;
        }}

        .monitoring-card {{
            padding: 18px;
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(14, 116, 144, 0.08), rgba(255, 255, 255, 0.94));
            border: 1px solid rgba(14, 116, 144, 0.16);
            display: grid;
            gap: 14px;
        }}

        .monitoring-header {{
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
        }}

        .monitoring-title {{
            display: grid;
            gap: 4px;
        }}

        .monitoring-title strong {{
            font-size: 1rem;
        }}

        .monitoring-title span {{
            font-size: 0.88rem;
            color: var(--muted);
        }}

        .refresh-pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(15, 118, 110, 0.1);
            color: var(--accent);
            font-weight: 700;
            font-size: 0.88rem;
        }}

        .monitor-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
        }}

        .mini-metric {{
            padding: 12px;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(19, 34, 56, 0.08);
        }}

        .mini-metric span {{
            display: block;
            font-size: 0.8rem;
            color: var(--muted);
            margin-bottom: 6px;
        }}

        .mini-metric strong {{
            font-size: 1.1rem;
        }}

        .metric-list {{
            display: grid;
            gap: 10px;
        }}

        .metric-row {{
            display: flex;
            justify-content: space-between;
            gap: 12px;
            font-size: 0.86rem;
            line-height: 1.4;
            color: #304153;
        }}

        .metric-row span:first-child {{
            color: var(--muted);
        }}

        .metric-panel {{
            padding: 12px;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(19, 34, 56, 0.08);
        }}

        .result-card {{
            padding: 18px;
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(248, 250, 252, 0.9));
            border: 1px solid var(--border);
        }}

        .result-label {{
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--muted);
            margin-bottom: 8px;
        }}

        .result-value {{
            font-size: 2.5rem;
            line-height: 1;
            margin: 0 0 8px;
            letter-spacing: -0.05em;
        }}

        .pill-row {{ display: flex; gap: 10px; flex-wrap: wrap; }}

        .pill {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(15, 118, 110, 0.1);
            color: var(--accent);
            font-weight: 700;
            font-size: 0.9rem;
        }}

        .message {{
            padding: 14px 16px;
            border-radius: 16px;
            background: rgba(18, 127, 93, 0.08);
            color: var(--success);
            border: 1px solid rgba(18, 127, 93, 0.18);
            min-height: 54px;
            line-height: 1.5;
            white-space: pre-wrap;
        }}

        .message.error {{
            background: rgba(180, 35, 24, 0.08);
            color: var(--danger);
            border-color: rgba(180, 35, 24, 0.18);
        }}

        .footer-note {{
            margin-top: 14px;
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.6;
        }}

        .status-dot {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }}

        .status-dot::before {{
            content: "";
            width: 10px;
            height: 10px;
            border-radius: 999px;
            background: #94a3b8;
        }}

        .status-dot.ok::before {{
            background: #16a34a;
        }}

        .status-dot.warn::before {{
            background: #d97706;
        }}

        .status-dot.error::before {{
            background: #dc2626;
        }}

        @media (max-width: 960px) {{
            .grid {{ grid-template-columns: 1fr; }}
            .result {{ position: static; }}
        }}

        @media (max-width: 720px) {{
            .shell {{ width: min(100% - 20px, 1180px); padding-top: 18px; }}
            .form-grid {{ grid-template-columns: 1fr; }}
            h1 {{ max-width: none; }}
        }}
    </style>
</head>
<body>
    <main class="shell">
        <section class="hero">
            <div>
                <div class="eyebrow">Telco churn prediction interface</div>
                <h1>Predict churn with a polished browser UI.</h1>
                <p class="lede">Fill in a customer profile, use the demo customer to get started, and see the churn probability instantly.</p>
            </div>
        </section>

        <section class="grid">
            <div class="panel">
                <div class="panel-header">
                    <h2>Customer inputs</h2>
                    <p>These fields match the training artifacts exactly. Categorical values are pulled from the saved encoders, so the form stays aligned with the model.</p>
                </div>
                <div class="panel-subhead">
                    <span class="step-chip">1. Try the demo</span>
                    <span class="step-chip">2. Edit only what you know</span>
                    <span class="step-chip">3. Review churn risk</span>
                </div>
                <form id="prediction-form">
                    <div id="field-grid" class="form-grid"></div>
                    <div class="actions">
                        <button class="primary" type="submit">Predict churn</button>
                        <button class="secondary" type="button" id="demo-button">Load demo customer</button>
                        <button class="ghost" type="button" id="reset-button">Reset form</button>
                    </div>
                </form>
            </div>

            <aside class="panel result">
                <div class="result-card">
                    <div class="result-label">Prediction</div>
                    <p class="result-value" id="probability">--</p>
                    <div class="pill-row">
                        <span class="pill" id="label-pill">Waiting for input</span>
                    </div>
                </div>
                <div class="message" id="message">Use the form to calculate churn risk.</div>
                <div class="monitoring-card">
                    <div class="monitoring-header">
                        <div class="monitoring-title">
                            <strong>Live monitoring</strong>
                            <span>Backend health, service stats, and runtime metrics.</span>
                        </div>
                        <span class="refresh-pill" id="monitor-refresh">Refreshing...</span>
                    </div>
                    <div class="monitor-grid">
                        <div class="mini-metric">
                            <span>Service status</span>
                            <strong id="monitor-health" class="status-dot warn">Checking</strong>
                        </div>
                        <div class="mini-metric">
                            <span>Total predictions</span>
                            <strong id="monitor-total">0</strong>
                        </div>
                        <div class="mini-metric">
                            <span>Positive rate</span>
                            <strong id="monitor-positive-rate">0%</strong>
                        </div>
                        <div class="mini-metric">
                            <span>Avg probability</span>
                            <strong id="monitor-avg-prob">0%</strong>
                        </div>
                    </div>
                    <div class="monitor-grid">
                        <div class="mini-metric">
                            <span>Latency (last request)</span>
                            <strong id="monitor-latency">--</strong>
                        </div>
                        <div class="mini-metric">
                            <span>Predicted positives</span>
                            <strong id="monitor-positives">0</strong>
                        </div>
                        <div class="mini-metric">
                            <span>CPU usage</span>
                            <strong id="monitor-cpu">--</strong>
                        </div>
                        <div class="mini-metric">
                            <span>Memory usage</span>
                            <strong id="monitor-memory">--</strong>
                        </div>
                    </div>
                    <div class="metric-panel">
                        <span class="result-label">Metric highlights</span>
                        <div class="metric-list">
                            <div class="metric-row"><span>Requests</span><span id="monitor-requests">--</span></div>
                            <div class="metric-row"><span>Latency avg</span><span id="monitor-latency-avg">--</span></div>
                            <div class="metric-row"><span>Error rate</span><span id="monitor-error-rate">--</span></div>
                        </div>
                    </div>
                </div>
                <div class="footer-note">If the API receives an unseen category or a missing field, the response will surface the validation error here. The monitoring card refreshes automatically.</div>
            </aside>
        </section>
    </main>

    <script id="field-data" type="application/json">{field_config_json}</script>
    <script id="section-data" type="application/json">{json.dumps([{"title": section["title"], "description": section["description"], "fields": sorted(section["fields"])} for section in FIELD_SECTIONS])}</script>
    <script>
        const fields = JSON.parse(document.getElementById("field-data").textContent);
        const sections = JSON.parse(document.getElementById("section-data").textContent);
        const fieldGrid = document.getElementById("field-grid");
        const form = document.getElementById("prediction-form");
        const probability = document.getElementById("probability");
        const labelPill = document.getElementById("label-pill");
        const message = document.getElementById("message");
        const demoButton = document.getElementById("demo-button");
        const resetButton = document.getElementById("reset-button");
        const monitorRefresh = document.getElementById("monitor-refresh");
        const monitorHealth = document.getElementById("monitor-health");
        const monitorTotal = document.getElementById("monitor-total");
        const monitorPositiveRate = document.getElementById("monitor-positive-rate");
        const monitorAvgProb = document.getElementById("monitor-avg-prob");
        const monitorLatency = document.getElementById("monitor-latency");
        const monitorPositives = document.getElementById("monitor-positives");
        const monitorRequests = document.getElementById("monitor-requests");
        const monitorLatencyAvg = document.getElementById("monitor-latency-avg");
        const monitorErrorRate = document.getElementById("monitor-error-rate");
        const monitorCpu = document.getElementById("monitor-cpu");
        const monitorMemory = document.getElementById("monitor-memory");

        function setStatusDot(element, state) {{
            element.classList.remove("ok", "warn", "error");
            if (state) {{
                element.classList.add(state);
            }}
        }}

        async function refreshMonitoring() {{
            monitorRefresh.textContent = "Refreshing...";

            try {{
                const [healthResponse, statsResponse] = await Promise.all([
                    fetch("/health"),
                    fetch("/stats"),
                ]);
                const systemResponse = await fetch("/system");

                if (healthResponse.ok) {{
                    monitorHealth.textContent = "Healthy";
                    setStatusDot(monitorHealth, "ok");
                }} else {{
                    monitorHealth.textContent = "Degraded";
                    setStatusDot(monitorHealth, "warn");
                }}

                if (statsResponse.ok) {{
                    const stats = await statsResponse.json();
                    monitorTotal.textContent = String(stats.predictions_total);
                    monitorPositiveRate.textContent = `${{(stats.positive_rate * 100).toFixed(1)}}%`;
                    monitorAvgProb.textContent = `${{(stats.average_churn_probability * 100).toFixed(1)}}%`;
                    monitorPositives.textContent = String(stats.predictions_positive);
                    monitorRequests.textContent = String(stats.predictions_total);
                    monitorLatencyAvg.textContent = "--";
                    monitorErrorRate.textContent = "--";
                }} else {{
                    monitorTotal.textContent = "--";
                    monitorPositiveRate.textContent = "--";
                    monitorAvgProb.textContent = "--";
                    monitorPositives.textContent = "--";
                    monitorRequests.textContent = "--";
                    monitorLatencyAvg.textContent = "--";
                    monitorErrorRate.textContent = "--";
                }}

                if (systemResponse.ok) {{
                    const system = await systemResponse.json();
                    monitorCpu.textContent = `${{system.cpu_percent.toFixed(1)}}%`;
                    monitorMemory.textContent = `${{system.memory_mb.toFixed(1)}} MB`;
                }} else {{
                    monitorCpu.textContent = "--";
                    monitorMemory.textContent = "--";
                }}

                const processTime = healthResponse.headers.get("X-Process-Time");
                monitorLatency.textContent = processTime ? `${{processTime}}s` : "--";
                monitorRefresh.textContent = `Updated ${{new Date().toLocaleTimeString()}}`;
            }} catch (error) {{
                monitorRefresh.textContent = "Monitoring unavailable";
                monitorHealth.textContent = "Offline";
                setStatusDot(monitorHealth, "error");
                monitorRequests.textContent = "--";
                monitorLatencyAvg.textContent = "--";
                monitorErrorRate.textContent = "--";
                monitorCpu.textContent = "--";
                monitorMemory.textContent = "--";
            }}
        }}

        function setMessage(text, isError = false) {{
            message.textContent = text;
            message.classList.toggle("error", isError);
        }}

        function renderField(field) {{
            const wrapper = document.createElement("div");
            wrapper.className = "field";

            const label = document.createElement("label");
            label.htmlFor = field.name;
            label.textContent = field.label;
            wrapper.appendChild(label);

            let control;
            if (field.kind === "select") {{
                control = document.createElement("select");
                field.options.forEach((option) => {{
                    const optionElement = document.createElement("option");
                    optionElement.value = option;
                    optionElement.textContent = option;
                    control.appendChild(optionElement);
                }});
                control.value = field.default;
            }} else {{
                control = document.createElement("input");
                control.type = "number";
                control.step = field.step;
                control.value = field.default;
            }}

            control.id = field.name;
            control.name = field.name;
            control.required = true;
            wrapper.appendChild(control);

            return wrapper;
        }}

        function renderSections() {{
            fieldGrid.innerHTML = "";

            for (const section of sections) {{
                const sectionBlock = document.createElement("div");
                sectionBlock.className = "section-block";

                const sectionHead = document.createElement("div");
                sectionHead.className = "section-head";

                const sectionTitle = document.createElement("h3");
                sectionTitle.textContent = section.title;
                sectionHead.appendChild(sectionTitle);

                const sectionDescription = document.createElement("p");
                sectionDescription.textContent = section.description;
                sectionHead.appendChild(sectionDescription);

                const sectionGrid = document.createElement("div");
                sectionGrid.className = "form-grid";

                section.fields.forEach((fieldName) => {{
                    const field = fields.find((item) => item.name === fieldName);
                    if (field) {{
                        sectionGrid.appendChild(renderField(field));
                    }}
                }});

                sectionBlock.appendChild(sectionHead);
                sectionBlock.appendChild(sectionGrid);
                fieldGrid.appendChild(sectionBlock);
            }}
        }}

        function applyFeatures(features) {{
            for (const field of fields) {{
                const control = document.getElementById(field.name);
                if (!control || !(field.name in features)) {{
                    continue;
                }}

                control.value = features[field.name];
            }}
        }}

        renderSections();
        refreshMonitoring();
        setInterval(refreshMonitoring, 15000);

        resetButton.addEventListener("click", () => {{
            form.reset();
            probability.textContent = "--";
            labelPill.textContent = "Waiting for input";
            setMessage("Use the form to calculate churn risk.");
        }});

        demoButton.addEventListener("click", () => {{
            applyFeatures({json.dumps(DEMO_FEATURES)});
            probability.textContent = "--";
            labelPill.textContent = "Demo loaded";
            setMessage("Demo customer loaded. Review the values and press Predict churn.");
        }});

        form.addEventListener("submit", async (event) => {{
            event.preventDefault();

            const features = {{}};
            for (const field of fields) {{
                const control = document.getElementById(field.name);
                features[field.name] = field.kind === "number" ? Number(control.value) : control.value;
            }}

            setMessage("Running prediction...");
            labelPill.textContent = "Processing";

            try {{
                const response = await fetch("/predict", {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: JSON.stringify({{ features }}),
                }});

                const data = await response.json();

                if (!response.ok) {{
                    const detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
                    throw new Error(detail || "Prediction failed.");
                }}

                probability.textContent = `${{(data.churn_probability * 100).toFixed(1)}}%`;
                labelPill.textContent = data.prediction === 1 ? "High churn risk" : "Low churn risk";
                setMessage(`Model output: churn_probability=${{data.churn_probability}}, prediction=${{data.prediction}}`);
            }} catch (error) {{
                probability.textContent = "--";
                labelPill.textContent = "Prediction error";
                setMessage(error.message, true);
            }}
        }});
    </script>
</body>
</html>"""
        return HTMLResponse(content=html)


@app.get("/health")
def health():
        return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/stats")
def stats():
    with STATS_LOCK:
        total = PREDICTION_COUNTS["positive"] + PREDICTION_COUNTS["negative"]
        average_probability = round(PREDICTION_SUM / total, 4) if total else 0.0
        positive_rate = round(PREDICTION_COUNTS["positive"] / total, 4) if total else 0.0

    return {
        "predictions_total": total,
        "predictions_positive": PREDICTION_COUNTS["positive"],
        "predictions_negative": PREDICTION_COUNTS["negative"],
        "positive_rate": positive_rate,
        "average_churn_probability": average_probability,
    }


@app.get("/system")
def system():
    cpu_percent = PROCESS.cpu_percent(interval=0.0)
    memory_mb = PROCESS.memory_info().rss / (1024 * 1024)
    return {
        "cpu_percent": round(cpu_percent, 2),
        "memory_mb": round(memory_mb, 2),
    }


@app.post("/predict")
def predict(record: CustomerRecord):
        data = record.features

        missing = [column for column in FEATURE_COLS if column not in data]
        if missing:
                raise HTTPException(status_code=422, detail=f"Missing fields: {missing}")

        row = pd.DataFrame([{column: data[column] for column in FEATURE_COLS}])

        for column, encoder in LABEL_ENCODERS.items():
                value = str(row.at[0, column])
                if value not in encoder.classes_:
                        raise HTTPException(
                                status_code=422,
                                detail=f"Unknown value '{value}' for field '{column}'. Valid values: {list(encoder.classes_)}",
                        )
                row[column] = encoder.transform([value])

        row[NUMERIC_COLS] = SCALER.transform(row[NUMERIC_COLS])

        probability = float(MODEL.predict_proba(row)[0][1])
        prediction = int(probability >= 0.5)

        PREDICTION_TOTAL.inc()
        with STATS_LOCK:
            global PREDICTION_SUM
            PREDICTION_SUM += probability
            if prediction == 1:
                PREDICTION_COUNTS["positive"] += 1
            else:
                PREDICTION_COUNTS["negative"] += 1

        return {"churn_probability": round(probability, 4), "prediction": prediction}
