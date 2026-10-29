# Model Update & Retraining Design

## Question
> **You receive 1,000 new labelled samples after deployment. Would your deployed model automatically retrain itself? Design the update process.**

---

## Direct Answer: No

**Under no circumstances should a deployed healthcare model retrain itself automatically.**

In a medical context like diabetes risk prediction:
1. **Patient Safety First:** Automatically changing a live model based on incoming data can silently reduce its accuracy, leading to missed diagnoses (false negatives) or wrong risk assessments.
2. **Data Quality Concerns:** A small batch of 1,000 samples might contain labeling mistakes, biased patient groups, or typing errors. Feeding it directly into a live model without checking can corrupt the model.
3. **Auditability & Traceability:** In medical software, every prediction must be traceable to an exact, unchanging model version. If a model changes by itself in the background, you cannot guarantee or reproduce past predictions.

Instead of automatic retraining, we use a **controlled, safe update process with human review and thorough testing**.

---

## Update Workflow Overview

```
[ 1,000 New Labelled Samples ]
              │
              ▼
    1. Data Quality Check (Check for missing values, invalid numbers, or bias)
              │
              ▼
    2. Offline Retraining (Trained separately, completely away from live users)
              │
              ▼
    3. Model Comparison (Test the New Model directly against the Current Model)
              │
              ▼
    4. Human Review & Approval (Clinical lead & engineer sign-off)
              │
              ▼
    5. Versioned Deployment (Deployed as v3 alongside v1/v2 — never overwritten)
              │
              ▼
    6. Live Monitoring & Instant Rollback (One-click switch back if any issue occurs)
```

---

## 1. Frozen Production Model (The Live Model Stays Untouched)

- **The live model never changes on the fly:** The running web service (`backend/app/main.py`) only reads model files from disk; it never modifies them.
- **Read-Only in Production:** The live server does not have training scripts or permission to overwrite models.
- **Reproducible Predictions:** Every prediction is saved in PostgreSQL with a unique fingerprint (SHA-256 hash) of the patient inputs, the exact model version used (`v1` or `v2`), and the time it was made. Any past prediction can be verified at any time.

---

## 2. Separate Offline Retraining Process

Retraining happens completely away from the live website and users:
1. **Save New Data:** The 1,000 new patient records are saved to a separate storage area.
2. **Check Data Quality:** Before training, the new data is checked:
   - Are all 21 patient fields present and within normal human limits (e.g., BMI between 10 and 100)?
   - Is the data skewed (for example, if all 1,000 records came from only one age group or clinic)?
3. **Separate Training Job:** Retraining runs as a separate background task on a different machine, without slowing down or touching the live website.
4. **Data Versioning:** The dataset used for training is saved with a version number so we always know which data was used to build which model.

---

## 3. Comparing the New Model Against the Current Model

Before any new model is considered for production, it must compete against the current live model on a fixed benchmark test set:

| What We Test | Goal for the New Model | Why It Matters |
| :--- | :--- | :--- |
| **Catching Diabetes (Recall / Sensitivity)** | Must match or beat Current Model (at least $82\%$) | Missing a sick patient is dangerous because they miss early treatment. |
| **Overall Accuracy** | Must match or beat Current Model | Ensures high diagnostic reliability without raising false alarms. |
| **Fairness Across Groups** | Must perform equally well across ages and genders | Prevents unfair bias toward any specific group of patients. |
| **Prediction Speed** | Under 50 milliseconds per request | Ensures doctors and apps get instant results without waiting. |

If the new model performs worse on **any** of these tests, it is rejected and the current live model continues running.

---

## 4. Human Review and Approval

Automated tests are important, but a human must always have the final decision in healthcare:
- **Test Summary Report:** The system generates a report showing accuracy graphs, confusion matrices, and test results.
- **Waiting Area (Staging):** The new model is saved in a draft state and cannot serve real traffic yet.

---

## 5. Versioned Deployment (Never Overwrite Existing Models)

When the new model is approved:
- **Never Overwrite Old Files:** The new model is saved as a new file (e.g., `v3.joblib`). The older versions (`v1.joblib` and `v2.keras`) remain untouched on disk.
- **Zero-Downtime Release:** The server loads the new file into memory while still serving traffic with the current model.
- **Safe Rollout (Test Run):**
  - **Silent Run (Shadow Mode):** The new model can run quietly in the background on live incoming requests to verify it runs smoothly without showing results to patients yet.
  - **Switch Traffic:** Once confirmed stable, the system switches active traffic over to the new version using the API (`POST /models/v3/activate`).

---

## 6. Instant Rollback Path

If any unexpected issue appears in production (e.g., higher latency, unexpected error rates, or doctor feedback):
1. **One-Command Immediate Rollback:** An operator can instantly switch traffic back to the previous proven model with a single command:
   ```bash
   curl -X POST http://localhost:8000/models/v1/activate
   ```
   The backend switches the active model in milliseconds **without restarting the server or dropping any requests**.
2. **Review Past Predictions Using Logs:** Because every single prediction is permanently logged in the PostgreSQL database with the model version, engineers can immediately view every patient prediction made by the problematic version:
   ```sql
   SELECT * FROM prediction_logs WHERE model_version = 'v3';
   ```
3. **Safe Demotion:** The problematic model is marked as `deprecated` so it cannot be accidentally activated again.

---

## Summary

| Requirement | How Our System Handles It |
| :--- | :--- |
| **Frozen Model** | Live server is read-only; no training happens in production. |
| **Offline Retraining** | Retraining happens in a separate background pipeline away from live traffic. |
| **Model Comparison** | The new model must match or beat the current live model on a benchmark test set. |
| **Human Approval** | A doctor / clinical lead and an engineer must review and approve before release. |
| **Versioned Deployment** | Models are saved with clear version numbers (`v1`, `v2`, `v3`) without overwriting. |
| **Instant Rollback** | Immediate one-click switch back to the previous version via API (`/models/{v}/activate`). |
