#  Diabetes Health Indicator — ML Inference Service

A **clean, production-style MLOps inference service** that converts the pretrained diabetes risk models from HuggingFace into a deployed REST API with PostgreSQL persistence, model versioning, and a React frontend.

---

##  Architecture & Stack

```
[ React Frontend ] ──(port 3000)──► [ FastAPI Service ] ──(port 8000)──► [ PostgreSQL ] ──(port 5433:5432)
                                             │
                                             ▼
                                     [ Models Directory ]
                                     ├── scaler.joblib
                                     ├── v1.joblib (RandomForest)
                                     └── v2.keras  (MLP Neural Net)
```

| Layer | Technology | Purpose |

| **Backend API** | FastAPI + Uvicorn |
| **ML Runtime** | Scikit-learn + Joblib + Keras | Model inference with standard feature scaler |
| **Database** | PostgreSQL | Persistent storage for predictions & model registry |
| **Frontend** | React | Interactive console for predictions, batch CSV & logs |
| **Containerization** | Docker + Docker Compose |
| **Testing** | pytest |



## 🚀 Quick Start (Docker Compose)

### 1. Configure environment
```bash
cp .env.example .env
```

### 2. Start all services
```bash
make docker-up
# or: docker compose up --build -d
```

This starts:
* **React Frontend**: http://localhost:3000
* **FastAPI Backend**: http://localhost:8000
* **Swagger API Docs**: http://localhost:8000/docs
* **PostgreSQL Database**: localhost:5433

---

## 🔌 API Endpoints

### 1. Inference
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/predict` | Single patient record JSON |
| `POST` | `/predict/batch` | Batch patient records JSON array |
| `POST` | `/predict/batch/csv` | Batch CSV multipart upload |

### 2. Model Management
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/models` | List all available model versions and active status |
| `POST` | `/models/{version}/activate` | Dynamically set the active model version |

### 3. History & Health
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/predictions` | Retrieve recent prediction audit logs |
| `GET` | `/predictions/{id}` | Retrieve single prediction by ID |
| `GET` | `/health` | Service health status and loaded models |

---

## Sample Request & Response

### Request (`POST /predict`)
```json
{
  "record": {
    "HighBP": 1, "HighChol": 0, "CholCheck": 1, "BMI": 28.5,
    "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0,
    "PhysActivity": 1, "Fruits": 1, "Veggies": 1,
    "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
    "GenHlth": 3, "MentHlth": 5, "PhysHlth": 2,
    "DiffWalk": 0, "Sex": 1, "Age": 7, "Education": 5, "Income": 6
  },
  "model_version": "v1"
}
```

### Response (`HTTP 200 OK`)
```json
{
  "id": "e81f7d4b-9e48-4389-bdc6-2c1b184766bb",
  "predicted_label": 1,
  "label": "Diabetes",
  "label_name": "diabetic",
  "confidence": 0.8732,
  "model_version": "v1",
  "latency_ms": 12,
  "metadata": {
    "model_version": "v1",
    "preprocessing_version": "1.0",
    "input_hash": "a1b2c3d4...",
    "timestamp": "2026-10-29T12:00:00Z"
  }
}
```

---

## 🧪 Testing

Run the automated test suite covering validation, model switching, single inference, and CSV upload:

```bash
make test
```


## 🗄️ Model Versioning & Logging Architecture

### 1. Model Versioning & Coexistence
- Multiple model artifacts coexist on disk (`models/v1.joblib` RandomForest, `models/v2.keras` MLP Neural Network, `models/scaler.joblib`).
- The active model version is tracked in PostgreSQL table `model_versions` and can be swapped dynamically without service downtime via `POST /models/{version}/activate`.
- Clients can either let the service use the default active model or explicitly pin a specific version per inference request (`"model_version": "v1"`).

### 2. Comprehensive Audit Logging
- Every single and batch prediction is persistently logged into the PostgreSQL `prediction_logs` table.
- Logged schema captures:
  - `id`: Unique prediction UUID.
  - `batch_id`: Batch UUID grouping CSV uploads (`NULL` for single form submissions).
  - `model_version`: Exact model artifact version utilized.
  - `input_payload`: Canonical JSON of input health indicators.
  - `input_hash`: Cryptographic SHA-256 hash for deterministic deduplication and audit verification.
  - `preprocessing_version`: Tracking data transform pipelines (e.g. `1.0`).
  - `predicted_label` & `confidence`: Diagnostic outcome and probability score.
  - `latency_ms`: Execution latency in milliseconds.
  - `status` & `error_message`: Tracks success (`success`) or inference failures (`failed`).
  - `created_at`: ISO-8601 UTC timestamp.


## 🩺 Model Evaluation, Promotion & Rollback Strategy

For the assignment requirement regarding **how a new model version is evaluated and promoted into production without silently replacing the currently deployed model** (see [DESIGN.md]for full clinical architecture):

### 1. Where New Models "Land" (Draft / Staging Area)
- Retrained models **never** deploy directly to production. Retraining runs in an isolated offline pipeline.
- New models land in a **Model Registry / storage area** tagged as a draft or staging version (e.g. `v3`).
- The live serving website continues using the current active model (`v1`) without interruption.

### 2. Validation & Evaluation Checks Before Promotion
Before any new model can go live, it must pass automated tests against a benchmark test dataset:
- **Catching Diabetes (Recall / Sensitivity $\ge 0.85$):** Prioritizes minimizing false negatives (undiagnosed patients).
- **Overall Accuracy:** New model must match or beat the current live model.
- **Fairness Across Patient Groups:** Accuracy must remain consistent across different age and sex groups.
- **Data Checks:** Incoming data is checked for bad numbers or demographic skew.
- **Human Approval:** Both a Clinical Specialist and an MLOps Engineer must review the test report before approval.

### 3. Operational Promotion & Instant Rollback
- **Test Run (Shadow Mode):** The new model can run silently in the background on live incoming requests to verify speed and stability without showing results to patients.
- **Atomic Promotion (Zero-Downtime Swap):** Promotion is done via a synchronized database switch (`POST /models/{version}/activate`). The server points to the new model instantly without restarting or dropping requests.
- **Instant Rollback Path:** If any issue occurs, an operator immediately runs `POST /models/{previous_version}/activate` to switch back to the previous proven model within milliseconds.
