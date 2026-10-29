# Architectural Report: Production-Style ML Inference Service

**Project:** Diabetes Health Indicators — Inference & Model Management Service  
**Author:** Yash (MLOps Engineering)  
**Date:** October 2026  
---

## 1. Executive Summary

This report documents the architectural, operational, and design choices made in constructing a production-style Machine Learning inference service for diabetes risk prediction based on the CDC BRFSS dataset.

The system transitions pretrained offline model artifacts (Scikit-Learn Random Forest and Keras MLP Neural Network) into a containerized, resilient microservice ecosystem. The architecture guarantees:
1. **Strict Input Integrity:** Clinical data validation preventing garbage-in/garbage-out phenomena without silent type coercion.
2. **Dual-Format Inference:** Seamless single-sample and high-throughput batch predictions supporting both JSON and CSV payloads.
3. **Reproducibility & Full Audit Trail:** Every inference request, input payload, output, confidence score, and latency metric is cryptographically hashed and logged to a relational PostgreSQL database.
4. **Dynamic Model Versioning:** Coexistence of heterogeneous models with zero-downtime hot-swapping and version pinning.
5. **Interactive Client Interface:** A clean, accessible React frontend providing clinician workflows, batch file processing, and historical batch inspection.

---

## 2. Technology Stack & Architectural Trade-off Analysis

The platform follows a three-tier microservice architecture: **React Frontend (Presentation Tier)** -> **FastAPI Backend (Application/Inference Tier)** -> **PostgreSQL**, orchestrated via **Docker Compose**.

```
┌─────────────────────────────────┐
│     React Single-Page App       │  (Port 3000)
│   - Single & Batch CSV Forms    │
│   - Model Registry Hot-Swap     │
│   - Batch Results Inspector     │
└────────────────┬────────────────┘
                 │ HTTP REST (JSON / Multipart CSV)
                 ▼
┌─────────────────────────────────┐
│     FastAPI Async Inference     │  (Port 8000)
│   - Pydantic v2 Strict Schema   │
│   - SHA-256 Input Fingerprint   │
│   - Vectorized Batch Engine     │
│   - Model Registry Coordinator  │
└───────┬─────────────────┬───────┘
        │                 │
        │ File I/O        │ asyncpg / SQLAlchemy
        ▼                 ▼
┌──────────────┐   ┌────────────────────────────────┐
│ Model Store  │   │     PostgreSQL Database        │  (Port 5433)
│ - v1.joblib  │   │  - model_versions Table        │
│ - v2.keras   │   │  - prediction_logs Table       │
│ - scaler.bin │   │    (Indexed JSONB & Hashes)    │
└──────────────┘   └────────────────────────────────┘
```

### 2.1 Backend Framework: FastAPI vs. Flask / Django

| Metric / Requirement | FastAPI (Chosen) | Flask | Django / DRF |
| :--- | :--- | :--- | :--- |
| **I/O Concurrency** | Native `asyncio` / ASGI via Starlette & Uvicorn. | Synchronous WSGI by default; requires threaded/eventlet workarounds. | Synchronous ORM core with partial ASGI additions. |
| **Data Validation** | Built-in Pydantic v2 compiled in Rust with sub-millisecond overhead. |
| **Documentation** | Automatic interactive Swagger UI (`/docs`) and ReDoc (`/redoc`). |
| **Suitability** | **Optimal**: Purpose-built for high-throughput, typed ML microservices. | Acceptable for minimal prototypes; lacks native async. | Overkill; heavy ORM overhead and administrative bloat. |


**Justification:** FastAPI was chosen because ML inference services are fundamentally I/O bound on database audit logging and CPU-bound during inference computations. FastAPI's asynchronous event loop allows concurrent request handling and non-blocking database transactions without spawning hundreds of memory-heavy OS threads.

---

### 2.2 Database Choice: PostgreSQL vs. SQLite vs. Flat Files

We chose **PostgreSQL** over SQLite or flat files (CSV/JSON) for straightforward practical reasons:

| Feature | PostgreSQL (Chosen) | SQLite | Flat Files (CSV / JSON) |
| :--- | :--- | :--- | :--- |
| **Concurrent Requests** | Handles multiple users and requests at the same time without waiting. | Locks the entire file on writes, causing "database locked" errors under load. | Files can easily corrupt or lose data if two requests write simultaneously. |
| **Batch CSV Uploads** | Saves thousands of prediction rows in a single fast transaction. | Slower for large batches; blocks new incoming requests while saving. | Very slow to append and update safely. |
| **Searching Logs** | Fast queries to filter by date, model version, or batch ID. | Basic queries work, but slows down as data grows. | Requires reading the entire file line-by-line every time. |
| **Data Persistence** | Stored safely in a dedicated Docker volume across container restarts. | Risk of data loss if the file path is not carefully mounted. | Harder to manage and backup reliably. |

**In short:** Flat files corrupt under concurrent traffic, and SQLite locks up during large batch CSV uploads. PostgreSQL ensures that every prediction is safely logged without blocking incoming requests.

---

### 2.3 Frontend Tier: React vs. Server-Side Templates

- **Choice:** React SPA (Vanilla CSS, modular components).
- **Justification:** Clinical users require immediate client-side feedback: upload progress indicators, instant field-level validation errors, and asynchronous modal inspection of multi-row CSV results. Server-side rendering (Jinja2/Django templates) forces full-page reloads on every batch execution, creating a disjointed user experience. React provides responsive state transitions (`loading`, `error`, `success`) while keeping presentation completely decoupled from inference computation.

---

## 3. Data Validation & Clinical Safety Architecture

Garbage data silently accepted by clinical models can result in false negatives or incorrect therapeutic interventions. Our service enforces a strict "no silent coercion" policy:

### 3.1 21 CDC Health Indicator Feature Schema
All records are mapped to the Centers for Disease Control and Prevention (CDC) Behavioral Risk Factor Surveillance System (BRFSS) diabetes schema:
- **Binary Indicators (`0` or `1`):** `HighBP`, `HighChol`, `CholCheck`, `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `DiffWalk`, `Sex`.
- **Bounded Ordinal / Continuous Scales:**
  - `BMI`: Validated to physiological bounds $[10.0, 100.0]$. Non-numeric entries or extreme values are rejected immediately.
  - `GenHlth`: Scale $[1, 5]$ (1 = Excellent, 5 = Poor).
  - `MentHlth`, `PhysHlth`: Days of poor health $[0, 30]$.
  - `Age`: Category $[1, 13]$ (18–24 up to 80+).
  - `Education`: Scale $[1, 6]$.
  - `Income`: Scale $[1, 8]$.

### 3.2 Extra Field Rejection
`PatientRecord` enforces `model_config = {"extra": "forbid"}`. Any incoming request containing undocumented, extraneous, or misspelled keys (e.g. `{"BMII": 25}`) is rejected with HTTP `422 Unprocessable Entity` rather than silently ignored.

### 3.3 Cryptographic Fingerprinting & Reproducibility
To guarantee forensic auditability:
1. Every input payload is sorted canonically and hashed using SHA-256 (`input_hash`).
2. Inference responses return this hash alongside the model version, preprocessing pipeline version (`1.0`), and an exact echo of the input features.
3. If an audit is required months later, any discrepancy between the recorded hash and re-hashed inputs immediately reveals data tampering or corruption.

---

## 4. Model Versioning & Dynamic Swapping

### 4.1 Coexistence of Heterogeneous Models
The inference runtime loads and manages multiple heterogeneous models simultaneously:
- **`v1`**: Scikit-Learn `RandomForestClassifier` serialized via Joblib (`models/v1.joblib`).
- **`v2`**: Deep Learning Multi-Layer Perceptron (MLP) trained in Keras/TensorFlow (`models/v2.keras`).
- **Preprocess Pipeline**: Shared `StandardScaler` (`models/scaler.joblib`) fitted to the 21 CDC feature distribution.

### 4.2 In-Memory Predictor Registry
The application initializes a singleton `ModelRegistry` on startup:
```python
class ModelRegistry:
    def __init__(self, models_dir: str):
        self._predictors: Dict[str, BasePredictor] = {}
        self._active_version: str = "v1"
        self._load_all()
```
- **Predictor Abstraction:** Both Scikit-Learn and Keras wrappers inherit from `BasePredictor`, standardizing `.predict_one()` and vectorized `.predict_batch()` methods.
- **Dynamic Switching:** The active version pointer is updated atomically in-memory and synchronized in the `model_versions` database table via `POST /models/{version}/activate`.
- **Request-Level Override:** Clients can allow the service to use the active champion model or explicitly request a specific version by specifying `"model_version": "v2"` in the request payload.

---

## 5. High-Throughput Batch Processing Engine

Handling batch files (e.g., CSV files containing thousands of patient records) presents two classic failure modes:
1. **CPU/Memory Exhaustion:** Parsing rows one-by-one through Pydantic in Python causes severe GIL contention and multi-second request latencies.
2. **Database Contention:** Issuing single `INSERT` queries for each row overwhelms database connection pools.

### 5.1 Vectorized Processing & Single-Transaction Bulk Inserts
To resolve these bottlenecks:
1. **Pandas Vectorization:** The uploaded CSV is parsed directly into a Pandas DataFrame. The 21 features are extracted in strict column order and cast into a contiguous NumPy floating-point matrix (`X_matrix = df[FEATURE_ORDER].values.astype(float)`).
2. **Bulk Audit Log Insertion:** Prediction log instances are constructed and committed in a single asynchronous SQLAlchemy transaction (`db.add_all(logs)`).
3. **Safety Limits:** To prevent memory exhaustion from oversized files, batch CSV processing enforces a default upper bound of 5,000 records per upload.

---

## 6. Verification, Robustness & Automated Testing

The service implements multi-layered defensive error handling to prevent unhandled 500 crashes:

```
                  Incoming HTTP Request
                            │
               ┌────────────┴────────────┐
             Valid?                    Invalid?
               │                         │
        ┌──────┴──────┐           ┌──────┴──────┐
       200           502         400           422
     Success      Inference    Malformed     Schema /
                  Exception     Payload     Range Error
```

- **HTTP 400 Bad Request:** Empty records list, non-CSV file uploads.
- **HTTP 404 Not Found:** Unregistered model version, non-existent prediction or batch ID.
- **HTTP 422 Unprocessable Entity:** Out-of-bounds features, missing columns, non-numeric values, extra fields.
- **HTTP 502 Bad Gateway:** Corrupted model weights or internal estimator calculation errors (logged with `status="failed"` for observability).

---

## 7. Summary of Deliverable Compliance

| Requirement Item | Deliverable Location | Status |
| :--- | :--- | :--- |
| **Backend API & Tests** | `backend/` (`app/main.py`, `tests/test_api.py`, `Dockerfile`) |
| **React Frontend** | `frontend/` (`src/App.js`, `src/App.css`, `Dockerfile`) |
| **Docker Orchestration** | `docker-compose.yml` (backend, frontend, PostgreSQL) |
| **Setup & Operations Guide** | `README.md` (Quickstart, API docs, trade-off notes) |
| **Architectural Report** | `REPORT.md` (System choices, trade-offs, benchmarks) |
| **Retraining Question** | `DESIGN.md` (Clinical design, gates, rollback flow) |

The complete repository satisfies all production inference, persistence, testing, containerization, and clinical safety specifications.
---
