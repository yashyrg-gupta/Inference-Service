import React, { useEffect, useState, useCallback } from "react";
import "./App.css";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

const FIELDS = [
  "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
  "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
  "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
  "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
];

const DEFAULT_RECORD = Object.fromEntries(
  FIELDS.map((f) => [f, f === "BMI" ? 25.0 : 0])
);


async function apiFetch(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    if (Array.isArray(body.detail)) {
      const errList = body.detail.map((d) => {
        const field = d.loc ? d.loc[d.loc.length - 1] : "Field";
        return `${field}: ${d.msg}`;
      });
      const err = new Error(errList.join(" | "));
      err.errors = errList;
      throw err;
    }
    const message =
      typeof body.detail === "string"
        ? body.detail
        : JSON.stringify(body.detail ?? body);
    throw new Error(message || `Request failed (${res.status})`);
  }
  return body;
}

function ErrorDisplay({ error }) {
  if (!error) return null;
  const errorItems = error.errors || [error.message || String(error)];
  return (
    <div className="error-box">
      <strong>Error:</strong>
      {errorItems.length === 1 ? (
        <span> {errorItems[0]}</span>
      ) : (
        <ul className="error-list">
          {errorItems.map((e, idx) => (
            <li key={idx}>{e}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function formatTimestamp(ts) {
  if (!ts) return "N/A";
  const iso =
    typeof ts === "string" && !ts.endsWith("Z") && !/[+-]\d{2}:\d{2}$/.test(ts)
      ? `${ts}Z`
      : ts;
  return new Date(iso).toLocaleString();
}

function ModelPanel({ models, onActivate }) {
  return (
    <section className="panel">
      <h2> Registered Models</h2>
      {models.length === 0 ? (
        <p className="muted">No models loaded yet.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Version</th>
              <th>Type</th>
              <th>Status</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.version}>
                <td><strong>{m.version}</strong></td>
                <td>{m.model_type || "Classifier"}</td>
                <td>
                  <span className={`badge ${m.is_active ? "badge-success" : "badge-secondary"}`}>
                    {m.is_active ? "Active" : "Inactive"}
                  </span>
                </td>
                <td>
                  {!m.is_active ? (
                    <button className="btn-small" onClick={() => onActivate(m.version)}>
                      Set Active
                    </button>
                  ) : ''}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

function SingleInferenceForm({ onResult, models }) {
  const [record, setRecord] = useState(DEFAULT_RECORD);
  const [selectedVersion, setSelectedVersion] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [singleResult, setSingleResult] = useState(null);
  const [showRaw, setShowRaw] = useState(false);

  const update = (field, value) => setRecord((r) => ({ ...r, [field]: value }));

  const submit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    setSingleResult(null);
    try {
      const payload = {
        record: Object.fromEntries(
          FIELDS.map((f) => [
            f,
            f === "BMI" ? parseFloat(record[f]) : parseInt(record[f], 10),
          ])
        ),
      };
      if (selectedVersion) {
        payload.model_version = selectedVersion;
      }
      const result = await apiFetch("/predict", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setSingleResult(result);
      onResult(result);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="panel">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.5rem" }}>
        <h2>Single Patient Prediction Form</h2>

      </div>

      <form onSubmit={submit} style={{ marginTop: "1rem" }}>
        <div style={{ marginBottom: "1rem" }}>
          <label style={{ fontWeight: 600 }}>
            Model Version:{" "}
            <select
              value={selectedVersion}
              onChange={(e) => setSelectedVersion(e.target.value)}
              style={{ marginLeft: "0.5rem", padding: "0.3rem 0.6rem" }}
            >
              <option value="">Default (Active Model)</option>
              {models.map((m) => (
                <option key={m.version} value={m.version}>
                  {m.version} {m.is_active ? "(Active)" : ""}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="grid-form">
          {FIELDS.map((field) => (
            <label key={field}>
              <span>{field}</span>
              <input
                type="number"
                step={field === "BMI" ? "0.1" : "1"}
                value={record[field]}
                onChange={(e) => update(field, e.target.value)}
                required
              />
            </label>
          ))}
        </div>

        <button type="submit" className="btn-primary" disabled={loading} style={{ marginTop: "1.25rem" }}>
          {loading ? "Running Inference..." : "Predict Single Sample"}
        </button>
      </form>

      <ErrorDisplay error={error} />

      {singleResult && (
        <div className="result-card">
          <div className="result-card-header">
            <div>
              <span className="muted" style={{ fontSize: "0.85rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                Diagnostic Prediction
              </span>
              <div style={{ marginTop: "0.25rem" }}>
                <span className={`result-badge ${singleResult.predicted_label === 1 ? "badge-danger" : "badge-success"}`}>
                  {singleResult.predicted_label === 1 ? "⚠️ Diabetic Risk Detected" : "✓ Non-Diabetic"}
                </span>
              </div>
            </div>
            <div style={{ textAlign: "right" }}>
              <span className="muted" style={{ fontSize: "0.8rem" }}>Confidence Score</span>
              <div style={{ fontSize: "1.3rem", fontWeight: 700 }}>
                {(singleResult.confidence * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          <div className="confidence-bar-container">
            <div
              className="confidence-bar-fill"
              style={{
                width: `${Math.min(100, singleResult.confidence * 100)}%`,
                background: singleResult.predicted_label === 1 ? "var(--danger)" : "var(--success)",
              }}
            />
          </div>

          <div className="result-metrics-grid">
            <div className="result-metric">
              <div className="label">Model Used</div>
              <div className="val">{singleResult.model_version}</div>
            </div>
            <div className="result-metric">
              <div className="label">Latency</div>
              <div className="val">{singleResult.latency_ms ?? 0} ms</div>
            </div>
            <div className="result-metric">
              <div className="label">Audit ID</div>
              <div className="val" style={{ fontSize: "0.75rem", fontFamily: "monospace" }}>
                {singleResult.id ? `${singleResult.id.slice(0, 8)}...` : "Saved"}
              </div>
            </div>
          </div>

          <div style={{ marginTop: "0.75rem" }}>
            <button
              type="button"
              className="btn-small"
              style={{ background: "#e2e8f0", color: "#334155" }}
              onClick={() => setShowRaw(!showRaw)}
            >
              {showRaw ? "Hide Raw Response" : "View Raw JSON & Hash Details"}
            </button>
            {showRaw && (
              <pre style={{ marginTop: "0.5rem", background: "#0f172a", color: "#38bdf8", padding: "0.75rem", borderRadius: "6px", fontSize: "0.8rem" }}>
                {JSON.stringify(singleResult, null, 2)}
              </pre>
            )}
          </div>
        </div>
      )}
    </section>
  );
}

function BatchCsvUpload({ onBatchResult, models }) {
  const [file, setFile] = useState(null);
  const [selectedVersion, setSelectedVersion] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [batchData, setBatchData] = useState(null);


  const submit = async (e) => {
    e.preventDefault();
    if (!file) return;
    setError(null);
    setLoading(true);
    setBatchData(null);
    try {
      const form = new FormData();
      form.append("file", file);
      if (selectedVersion) {
        form.append("model_version", selectedVersion);
      }
      const res = await fetch(`${API_BASE}/predict/batch/csv`, {
        method: "POST",
        body: form,
      });
      const body = await res.json();
      if (!res.ok) {
        if (Array.isArray(body.detail)) {
          const err = new Error(body.detail.map((d) => d.msg || JSON.stringify(d)).join(" | "));
          err.errors = body.detail.map((d) => d.msg || JSON.stringify(d));
          throw err;
        }
        throw new Error(
          typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)
        );
      }
      setBatchData(body);
      onBatchResult(body);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="panel">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.5rem" }}>
        <h2>Batch Prediction (CSV Upload)</h2>
      </div>

      <p className="muted" style={{ margin: "0.5rem 0 1rem 0" }}>
        Upload a CSV containing patient feature columns to run vectorized inference.
      </p>

      <form onSubmit={submit} className="upload-form">
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files[0])}
          style={{ flex: 1, minWidth: "220px" }}
        />
        <select
          value={selectedVersion}
          onChange={(e) => setSelectedVersion(e.target.value)}
          style={{ padding: "0.5rem", borderRadius: "4px", border: "1px solid var(--border)" }}
        >
          <option value="">Default (Active Model)</option>
          {models.map((m) => (
            <option key={m.version} value={m.version}>
              {m.version} {m.is_active ? "(Active)" : ""}
            </option>
          ))}
        </select>
        <button type="submit" className="btn-primary" disabled={loading || !file}>
          {loading ? "Uploading & Predicting..." : "Run Batch Inference"}
        </button>
      </form>

      <ErrorDisplay error={error} />

      {batchData && (
        <div style={{ marginTop: "1.25rem", borderTop: "1px solid var(--border)", paddingTop: "1rem" }}>
          <h3>Batch Execution Completed</h3>
          <p>
            <strong>Batch ID:</strong> <code>{batchData.batch_id}</code> |{" "}
            <strong>Rows Processed:</strong> {batchData.count}{" "}
            {batchData.total_rows_in_file > batchData.count ? `(out of ${batchData.total_rows_in_file} in file)` : ""}
          </p>
          <div style={{ display: "flex", gap: "1rem", marginBottom: "1rem" }}>
            <span className="badge badge-danger" style={{ fontSize: "0.9rem", padding: "0.3rem 0.75rem" }}>
              Diabetic Risk: {batchData.summary?.diabetic ?? 0}
            </span>
            <span className="badge badge-success" style={{ fontSize: "0.9rem", padding: "0.3rem 0.75rem" }}>
              Non-Diabetic: {batchData.summary?.non_diabetic ?? 0}
            </span>
          </div>

          <h4>Preview Results (First {Math.min(batchData.results.length, 10)} records):</h4>
          <table>
            <thead>
              <tr>
                <th>Row</th>
                <th>Result</th>
                <th>Confidence</th>
                <th>Model</th>
              </tr>
            </thead>
            <tbody>
              {batchData.results.slice(0, 10).map((r) => (
                <tr key={r.id}>
                  <td>#{r.row}</td>
                  <td>
                    <span className={`badge ${r.predicted_label === 1 ? "badge-danger" : "badge-success"}`}>
                      {r.label_name || (r.predicted_label === 1 ? "Diabetic" : "Non-Diabetic")}
                    </span>
                  </td>
                  <td>{(r.confidence * 100).toFixed(1)}%</td>
                  <td><span className="badge badge-secondary">{r.model_version}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function BatchDetailModal({ batchId, onClose }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!batchId) return;
    setLoading(true);
    apiFetch(`/batches/${batchId}`)
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [batchId]);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Batch Results Inspector</h3>
          <button className="btn-close" onClick={onClose}>✕</button>
        </div>

        <div className="modal-body">
          {loading && <p>Loading batch records...</p>}
          {error && <div className="error-box">{error}</div>}

          {data && (
            <div>
              <p>
                <strong>Batch ID:</strong> <code>{data.batch_id}</code><br />
                <strong>Model:</strong> <span className="badge badge-secondary">{data.model_version}</span> |{" "}
                <strong>Timestamp:</strong> {formatTimestamp(data.created_at)}
              </p>

              <div style={{ display: "flex", gap: "1rem", margin: "1rem 0" }}>
                <span className="badge badge-secondary" style={{ padding: "0.3rem 0.6rem" }}>
                  Total: {data.count}
                </span>
                <span className="badge badge-danger" style={{ padding: "0.3rem 0.6rem" }}>
                  Diabetic: {data.summary?.diabetic ?? 0}
                </span>
                <span className="badge badge-success" style={{ padding: "0.3rem 0.6rem" }}>
                  Non-Diabetic: {data.summary?.non_diabetic ?? 0}
                </span>
              </div>

              <table>
                <thead>
                  <tr>
                    <th>Row</th>
                    <th>Prediction</th>
                    <th>Confidence</th>
                    <th>Model</th>
                  </tr>
                </thead>
                <tbody>
                  {data.results.map((r) => (
                    <tr key={r.id}>
                      <td>#{r.row}</td>
                      <td>
                        <span className={`badge ${r.predicted_label === 1 ? "badge-danger" : "badge-success"}`}>
                          {r.label_name || (r.predicted_label === 1 ? "Diabetic" : "Non-Diabetic")}
                        </span>
                      </td>
                      <td>{(r.confidence * 100).toFixed(1)}%</td>
                      <td><span className="badge badge-secondary">{r.model_version}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="modal-footer">
          <button className="btn-primary" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}

function HistoryAndBatches({
  predictions,
  batches,
  activeTab,
  setActiveTab,
  onInspectBatch,
}) {
  return (
    <section className="panel">
      <h2>Inference Logs & Uploaded CSV History</h2>

      <div className="tab-bar">
        <button
          className={`tab-btn ${activeTab === "all" ? "active" : ""}`}
          onClick={() => setActiveTab("all")}
        >
          All Inferences ({predictions.length})
        </button>
        <button
          className={`tab-btn ${activeTab === "single" ? "active" : ""}`}
          onClick={() => setActiveTab("single")}
        >
          Single Sample Form Inferences
        </button>
        <button
          className={`tab-btn ${activeTab === "batches" ? "active" : ""}`}
          onClick={() => setActiveTab("batches")}
        >
          Uploaded CSV Batches ({batches.length})
        </button>
      </div>

      {activeTab === "batches" ? (
        batches.length === 0 ? (
          <p className="muted">No CSV batches uploaded yet.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Date / Time</th>
                <th>Batch ID</th>
                <th>Model</th>
                <th>Rows Processed</th>
                <th>Breakdown (Diabetic / Non-Diabetic)</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {batches.map((b) => (
                <tr key={b.batch_id}>
                  <td>{formatTimestamp(b.created_at)}</td>
                  <td><code>{b.batch_id ? b.batch_id.slice(0, 8) : "N/A"}...</code></td>
                  <td><span className="badge badge-secondary">{b.model_version}</span></td>
                  <td><strong>{b.count}</strong> rows</td>
                  <td>
                    <span className="badge badge-danger" style={{ marginRight: "0.5rem" }}>
                      {b.diabetic} Diabetic
                    </span>
                    <span className="badge badge-success">
                      {b.non_diabetic} Healthy
                    </span>
                  </td>
                  <td>
                    <button
                      className="btn-small"
                      onClick={() => onInspectBatch(b.batch_id)}
                    >
                      View Results
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )
      ) : predictions.length === 0 ? (
        <p className="muted">No predictions logged yet.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Type</th>
              <th>Model</th>
              <th>Result</th>
              <th>Confidence</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((p) => (
              <tr key={p.id}>
                <td>{formatTimestamp(p.created_at)}</td>
                <td>
                  <span className={`badge ${p.batch_id ? "badge-secondary" : "badge-secondary"}`}>
                    {p.batch_id ? "Batch CSV" : "Single Sample"}
                  </span>
                </td>
                <td><span className="badge badge-secondary">{p.model_version}</span></td>
                <td>
                  <span className={`badge ${p.predicted_label === 1 || p.label === "Diabetes" || p.label_name === "diabetic" ? "badge-danger" : "badge-success"}`}>
                    {p.label_name || p.label || (p.predicted_label === 1 ? "Diabetic" : "Non-Diabetic")}
                  </span>
                </td>
                <td>{(p.confidence * 100).toFixed(1)}%</td>
                <td><span className="badge badge-secondary">{p.status || "success"}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

export default function App() {
  const [models, setModels] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [batches, setBatches] = useState([]);
  const [historyTab, setHistoryTab] = useState("all");
  const [selectedBatchId, setSelectedBatchId] = useState(null);
  const [globalError, setGlobalError] = useState(null);

  const refresh = useCallback(async (tab = historyTab) => {
    try {
      const predEndpoint = tab === "single"
        ? "/predictions?limit=50&kind=single"
        : "/predictions?limit=50";

      const [m, p, b] = await Promise.all([
        apiFetch("/models"),
        apiFetch(predEndpoint).catch(() => []),
        apiFetch("/batches").catch(() => []),
      ]);

      setModels(Array.isArray(m) ? m : []);
      setPredictions(Array.isArray(p) ? p : []);
      setBatches(Array.isArray(b) ? b : []);
    } catch (err) {
      setGlobalError(err.message);
    }
  }, [historyTab]);

  useEffect(() => {
    refresh(historyTab);
  }, [refresh, historyTab]);

  const activateModel = async (version) => {
    try {
      await apiFetch(`/models/${version}/activate`, { method: "POST" });
      refresh();
    } catch (err) {
      setGlobalError(err.message);
    }
  };

  const handleInferenceDone = () => {
    refresh();
  };

  return (
    <main className="app-container">
      <header className="app-header">
        <h1> Diabetes Health Indicators — Inference Console</h1>

      </header>

      {globalError && <div className="error-box">{globalError}</div>}

      <ModelPanel models={models} onActivate={activateModel} />
      <SingleInferenceForm onResult={handleInferenceDone} models={models} />
      <BatchCsvUpload onBatchResult={handleInferenceDone} models={models} />

      <HistoryAndBatches
        predictions={predictions}
        batches={batches}
        activeTab={historyTab}
        setActiveTab={setHistoryTab}
        onInspectBatch={(bId) => setSelectedBatchId(bId)}
      />

      {selectedBatchId && (
        <BatchDetailModal
          batchId={selectedBatchId}
          onClose={() => setSelectedBatchId(null)}
        />
      )}
    </main>
  );
}
