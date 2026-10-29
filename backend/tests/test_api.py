import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app
from app.loader import ModelInferenceError, registry

VALID_RECORD = {
    "HighBP": 1, "HighChol": 0, "CholCheck": 1, "BMI": 28.5,
    "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0,
    "PhysActivity": 1, "Fruits": 1, "Veggies": 1,
    "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
    "GenHlth": 3, "MentHlth": 5, "PhysHlth": 2,
    "DiffWalk": 0, "Sex": 1, "Age": 7, "Education": 5, "Income": 6,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── 1. Health & Service Status ────────────────────────────────────────────────
def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "active_model_version" in data
    assert data["database"] == "ok"


# ── 2. Valid Single Inference (JSON & CSV) ────────────────────────────────────
def test_predict_single_json_valid(client):
    resp = client.post("/predict", json={"record": VALID_RECORD})
    assert resp.status_code == 200
    data = resp.json()
    assert data["predicted_label"] in (0, 1)
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["label"] in ("Diabetes", "No Diabetes")
    assert "metadata" in data
    assert data["metadata"]["model_version"]
    assert data["metadata"]["preprocessing_version"] == "1.0"
    assert data["metadata"]["input_hash"]
    assert data["metadata"]["input_echo"] == VALID_RECORD
    assert data["metadata"]["timestamp"]


def test_predict_single_csv_valid(client):
    headers = list(VALID_RECORD.keys())
    row = [str(VALID_RECORD[h]) for h in headers]
    csv_bytes = (",".join(headers) + "\n" + ",".join(row) + "\n").encode("utf-8")
    files = {"file": ("single_patient.csv", io.BytesIO(csv_bytes), "text/csv")}

    resp = client.post("/predict/csv", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["predicted_label"] in (0, 1)
    assert 0.0 <= data["confidence"] <= 1.0
    assert "metadata" in data
    assert data["metadata"]["input_echo"]


# ── 3. Valid Batch Inference (JSON & CSV) ──────────────────────────────────────
def test_predict_batch_json_valid(client):
    resp = client.post("/predict/batch", json={"records": [VALID_RECORD, VALID_RECORD]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert len(data["results"]) == 2
    assert "batch_id" in data
    for item in data["results"]:
        assert item["predicted_label"] in (0, 1)
        assert 0.0 <= item["confidence"] <= 1.0


def test_predict_batch_csv_valid(client):
    headers = list(VALID_RECORD.keys())
    row1 = [str(VALID_RECORD[h]) for h in headers]
    csv_bytes = (",".join(headers) + "\n" + ",".join(row1) + "\n" + ",".join(row1) + "\n").encode("utf-8")
    files = {"file": ("batch_patients.csv", io.BytesIO(csv_bytes), "text/csv")}

    resp = client.post("/predict/batch/csv", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert "batch_id" in data
    assert "summary" in data
    assert "diabetic" in data["summary"]
    assert "non_diabetic" in data["summary"]


# ── 4. Input Validation & Range Checks ────────────────────────────────────────
def test_validation_bmi_out_of_range(client):
    # BMI < 10.0 rejected
    bad_record = {**VALID_RECORD, "BMI": 5.0}
    resp = client.post("/predict", json={"record": bad_record})
    assert resp.status_code == 422

    # BMI > 100.0 rejected
    bad_record_high = {**VALID_RECORD, "BMI": 105.0}
    resp2 = client.post("/predict", json={"record": bad_record_high})
    assert resp2.status_code == 422


def test_validation_binary_field_out_of_range(client):
    # HighBP must be 0 or 1
    bad_record = {**VALID_RECORD, "HighBP": 2}
    resp = client.post("/predict", json={"record": bad_record})
    assert resp.status_code == 422


def test_validation_general_health_range(client):
    # GenHlth must be 1 to 5
    bad_record = {**VALID_RECORD, "GenHlth": 0}
    resp = client.post("/predict", json={"record": bad_record})
    assert resp.status_code == 422

    bad_record_high = {**VALID_RECORD, "GenHlth": 6}
    resp2 = client.post("/predict", json={"record": bad_record_high})
    assert resp2.status_code == 422


def test_validation_wrong_type(client):
    bad_record = {**VALID_RECORD, "BMI": "not_a_number"}
    resp = client.post("/predict", json={"record": bad_record})
    assert resp.status_code == 422


# ── 5. Missing & Unexpected Fields ────────────────────────────────────────────
def test_validation_missing_field(client):
    bad_record = {k: v for k, v in VALID_RECORD.items() if k != "BMI"}
    resp = client.post("/predict", json={"record": bad_record})
    assert resp.status_code == 422


def test_validation_unexpected_field_forbidden(client):
    # extra="forbid" rejects unexpected extra fields
    bad_record = {**VALID_RECORD, "UnexpectedField": 999}
    resp = client.post("/predict", json={"record": bad_record})
    assert resp.status_code == 422


# ── 6. Malformed & Empty Inputs ───────────────────────────────────────────────
def test_predict_batch_empty_rejected(client):
    resp = client.post("/predict/batch", json={"records": []})
    assert resp.status_code in (400, 422)


def test_predict_malformed_csv_rejected(client):
    bad_csv = b"random,garbage,not,the,brfss,columns\n1,2,3,4,5,6\n"
    files = {"file": ("bad.csv", io.BytesIO(bad_csv), "text/csv")}
    resp = client.post("/predict/batch/csv", files=files)
    assert resp.status_code == 422


def test_predict_non_csv_extension_rejected(client):
    files = {"file": ("test.txt", io.BytesIO(b"some text"), "text/plain")}
    resp = client.post("/predict/batch/csv", files=files)
    assert resp.status_code == 422


# ── 7. Model Version Loading & Selection ──────────────────────────────────────
def test_models_list_and_activate(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    models = resp.json()
    assert len(models) >= 2
    versions = [m["version"] for m in models]
    assert "v1" in versions
    assert "v2" in versions

    # Activate v2
    resp_v2 = client.post("/models/v2/activate")
    assert resp_v2.status_code == 200
    assert resp_v2.json()["version"] == "v2"
    assert resp_v2.json()["is_active"] is True

    # Revert to v1
    resp_v1 = client.post("/models/v1/activate")
    assert resp_v1.status_code == 200
    assert resp_v1.json()["version"] == "v1"


def test_predict_with_explicit_version(client):
    resp_v1 = client.post("/predict", json={"record": VALID_RECORD, "model_version": "v1"})
    assert resp_v1.status_code == 200
    assert resp_v1.json()["model_version"] == "v1"

    resp_v2 = client.post("/predict", json={"record": VALID_RECORD, "model_version": "v2"})
    assert resp_v2.status_code == 200
    assert resp_v2.json()["model_version"] == "v2"


def test_predict_with_invalid_version(client):
    resp = client.post("/predict", json={"record": VALID_RECORD, "model_version": "v999"})
    assert resp.status_code in (404, 422)


# ── 8. Model Inference Failure Handling ────────────────────────────────────────
def test_model_inference_failure_handling(client):
    predictor = registry.get("v1")
    with patch.object(predictor, "predict_one", side_effect=ModelInferenceError("Simulated hardware error")):
        resp = client.post("/predict", json={"record": VALID_RECORD, "model_version": "v1"})
        assert resp.status_code == 502
        assert "Model inference failed" in resp.json()["detail"]


# ── 9. Persistence of Inference Results & Logs ────────────────────────────────
def test_inference_persistence_and_retrieval(client):
    pred_resp = client.post("/predict", json={"record": VALID_RECORD})
    assert pred_resp.status_code == 200
    pred_id = pred_resp.json()["id"]

    hist_resp = client.get("/predictions?limit=20&kind=single")
    assert hist_resp.status_code == 200
    logs = hist_resp.json()
    ids = [l["id"] for l in logs]
    assert pred_id in ids

    single_resp = client.get(f"/predictions/{pred_id}")
    assert single_resp.status_code == 200
    record_data = single_resp.json()
    assert record_data["id"] == pred_id
    assert record_data["input_payload"] == VALID_RECORD
    assert record_data["input_hash"]
    assert record_data["status"] == "success"


def test_batch_persistence_and_retrieval(client):
    headers = list(VALID_RECORD.keys())
    row = [str(VALID_RECORD[h]) for h in headers]
    csv_bytes = (",".join(headers) + "\n" + ",".join(row) + "\n").encode("utf-8")
    files = {"file": ("batch_verify.csv", io.BytesIO(csv_bytes), "text/csv")}

    batch_resp = client.post("/predict/batch/csv", files=files)
    assert batch_resp.status_code == 200
    batch_id = batch_resp.json()["batch_id"]

    batches_resp = client.get("/batches")
    assert batches_resp.status_code == 200
    batch_ids = [b["batch_id"] for b in batches_resp.json()]
    assert batch_id in batch_ids

    details_resp = client.get(f"/batches/{batch_id}")
    assert details_resp.status_code == 200
    assert details_resp.json()["batch_id"] == batch_id
    assert len(details_resp.json()["results"]) >= 1


# ── 10. API Error Handling (404s) ─────────────────────────────────────────────
def test_get_nonexistent_prediction_404(client):
    resp = client.get("/predictions/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Prediction not found"


def test_get_nonexistent_batch_404(client):
    resp = client.get("/batches/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Batch not found"


# ── 11. Robustness & Fault Tolerance ──────────────────────────────────────────
def test_predict_empty_csv_file_rejected(client):
    files = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    resp = client.post("/predict/batch/csv", files=files)
    assert resp.status_code == 422
    assert "CSV" in resp.json()["detail"]


def test_predict_csv_missing_columns_rejected(client):
    incomplete_csv = b"HighBP,HighChol\n1,0\n"
    files = {"file": ("incomplete.csv", io.BytesIO(incomplete_csv), "text/csv")}
    resp = client.post("/predict/batch/csv", files=files)
    assert resp.status_code == 422
    assert "missing required columns" in resp.json()["detail"].lower()


def test_predict_batch_csv_non_numeric_rejected(client):
    headers = list(VALID_RECORD.keys())
    bad_row = [str(VALID_RECORD[h]) for h in headers]
    bad_row[headers.index("BMI")] = "not_a_float"
    bad_csv = (",".join(headers) + "\n" + ",".join(bad_row) + "\n").encode("utf-8")
    files = {"file": ("bad_type.csv", io.BytesIO(bad_csv), "text/csv")}

    resp = client.post("/predict/batch/csv", files=files)
    assert resp.status_code == 422
    assert "non-numeric" in resp.json()["detail"].lower() or "invalid" in resp.json()["detail"].lower()


def test_batch_model_inference_failure_handling(client):
    predictor = registry.get("v1")
    with patch.object(predictor, "predict_batch", side_effect=ModelInferenceError("Simulated batch failure")):
        headers = list(VALID_RECORD.keys())
        row = [str(VALID_RECORD[h]) for h in headers]
        csv_bytes = (",".join(headers) + "\n" + ",".join(row) + "\n").encode("utf-8")
        files = {"file": ("batch.csv", io.BytesIO(csv_bytes), "text/csv")}

        resp = client.post("/predict/batch/csv", files=files)
        assert resp.status_code == 502
        assert "Model inference failed" in resp.json()["detail"]

