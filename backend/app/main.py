import hashlib,io,json,time,uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List

import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select, update, text, func, case, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import async_engine, Base, get_db
from app.models import ModelVersion, PredictionLog
from app.loader import registry, FEATURE_ORDER, ModelLoadError, ModelInferenceError
from app.schemas import (
    SinglePredictRequest,
    BatchPredictRequest,
    PatientRecord,
)

settings = get_settings()
LABEL_NAMES = {0: "non-diabetic", 1: "diabetic"}


def hash_record(record_dict: dict) -> str:
    canonical = json.dumps(record_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def format_iso(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    if dt.tzinfo is None:
        return f"{dt.isoformat()}Z"
    return dt.isoformat()


# ── Lifespan Startup & Shutdown ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-create database tables
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Seed model registry in database if empty
    async with AsyncSession(async_engine) as session:
        result = await session.execute(select(ModelVersion))
        existing = result.scalars().all()
        if not existing:
            available = registry.list_available_versions() or ["v1", "v2"]
            for i, v in enumerate(available):
                mv = ModelVersion(
                    version=v,
                    model_type="RandomForest" if v == "v1" else "MLP (Keras)",
                    artifact_path=f"models/{v}",
                    is_active=(i == 0),
                )
                session.add(mv)
            await session.commit()

    yield

    await async_engine.dispose()


app = FastAPI(
    title="Diabetes Health Indicators — Inference Service",
    description="Clean, production-style inference service for diabetes risk prediction.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health_check(db: AsyncSession = Depends(get_db)):
    db_status = "ok"
    try:
        await db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"error: {e}"

    return {
        "status": "ok",
        "active_model_version": registry.active_version,
        "models_loaded": registry.list_available_versions(),
        "database": db_status,
    }


# ── Inference Helper ──────────────────────────────────────────────────────────
async def _execute_single(record: dict, version: Optional[str], db: AsyncSession, batch_id: Optional[str] = None):
    used_version = version or registry.active_version
    input_h = hash_record(record)
    start_time = time.perf_counter()

    try:
        predictor = registry.get(used_version)
        label, confidence = predictor.predict_one(record)
    except (ModelLoadError, ModelInferenceError, KeyError) as exc:
        log = PredictionLog(
            batch_id=batch_id,
            model_version=used_version,
            input_payload=record,
            input_hash=input_h,
            predicted_label=-1,
            confidence=0.0,
            status="failed",
            error_message=str(exc),
        )
        db.add(log)
        await db.commit()
        raise HTTPException(status_code=502, detail=f"Model inference failed: {exc}")

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    log = PredictionLog(
        batch_id=batch_id,
        model_version=used_version,
        input_payload=record,
        input_hash=input_h,
        preprocessing_version="1.0",
        predicted_label=label,
        confidence=round(confidence, 6),
        latency_ms=latency_ms,
        status="success",
    )
    db.add(log)
    await db.commit()

    return {
        "id": log.id,
        "predicted_label": label,
        "label": "Diabetes" if label == 1 else "No Diabetes",
        "label_name": LABEL_NAMES.get(label, "unknown"),
        "confidence": round(confidence, 6),
        "model_version": used_version,
        "latency_ms": latency_ms,
        "metadata": {
            "model_version": used_version,
            "preprocessing_version": "1.0",
            "input_hash": input_h,
            "input_echo": record,
            "timestamp": format_iso(datetime.utcnow()),
        },
    }


# ── Prediction Endpoints ──────────────────────────────────────────────────────
@app.post("/predict", tags=["Inference"])
async def predict_single(request: SinglePredictRequest, db: AsyncSession = Depends(get_db)):
    record_dict = request.record.model_dump()
    return await _execute_single(record_dict, request.model_version, db)


@app.post("/predict/csv", tags=["Inference"])
async def predict_single_csv(
    file: UploadFile = File(...),
    model_version: Optional[str] = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Accept a single patient record uploaded as a CSV file."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=422, detail="Only .csv files are accepted")

    content = await file.read()
    try:
        df = pd.read_csv(io.StringIO(content.decode("utf-8-sig", errors="ignore")))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Malformed CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=422, detail="CSV file contains no data rows")

    row_dict = df.iloc[0].to_dict()
    try:
        validated_record = PatientRecord(**row_dict)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return await _execute_single(validated_record.model_dump(), model_version, db)


@app.post("/predict/batch", tags=["Inference"])
async def predict_batch(request: BatchPredictRequest, db: AsyncSession = Depends(get_db)):
    if not request.records:
        raise HTTPException(status_code=400, detail="records list cannot be empty")
    batch_id = str(uuid.uuid4())
    results = []
    for rec in request.records:
        r = await _execute_single(rec.model_dump(), request.model_version, db, batch_id=batch_id)
        results.append(r)
    return {
        "batch_id": batch_id,
        "count": len(results),
        "results": results,
    }


@app.post("/predict/batch/csv", tags=["Inference"])
async def predict_batch_csv(
    file: UploadFile = File(...),
    model_version: Optional[str] = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=422, detail="Only .csv files are accepted")

    content = await file.read()
    try:
        df = pd.read_csv(io.StringIO(content.decode("utf-8-sig", errors="ignore")))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Malformed CSV: {e}")

    # Validate columns
    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"CSV missing required columns: {missing}")

    used_version = model_version or registry.active_version
    try:
        predictor = registry.get(used_version)
    except ModelLoadError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if df.empty:
        raise HTTPException(status_code=422, detail="CSV file contains no data rows")

    total_in_file = len(df)
    MAX_BATCH_ROWS = 5000
    if total_in_file > MAX_BATCH_ROWS:
        df = df.iloc[:MAX_BATCH_ROWS]

    # Vectorized fast inference with graceful error handling
    try:
        X_matrix = df[FEATURE_ORDER].values.astype(float)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"CSV contains non-numeric or invalid data: {exc}")

    try:
        labels, confidences = predictor.predict_batch(X_matrix)
    except (ModelInferenceError, Exception) as exc:
        raise HTTPException(status_code=502, detail=f"Model inference failed: {exc}")

    batch_id = str(uuid.uuid4())
    logs = []
    results = []

    for idx, (lbl, conf) in enumerate(zip(labels, confidences)):
        row_dict = df.iloc[idx][FEATURE_ORDER].to_dict()
        input_h = hash_record(row_dict)
        pred_id = str(uuid.uuid4())

        log = PredictionLog(
            id=pred_id,
            batch_id=batch_id,
            model_version=used_version,
            input_payload=row_dict,
            input_hash=input_h,
            preprocessing_version="1.0",
            predicted_label=int(lbl),
            confidence=round(float(conf), 4),
            latency_ms=0,
            status="success",
        )
        logs.append(log)

        results.append({
            "id": pred_id,
            "row": idx + 1,
            "predicted_label": int(lbl),
            "label": "Diabetes" if lbl == 1 else "No Diabetes",
            "label_name": LABEL_NAMES.get(int(lbl), "unknown"),
            "confidence": round(float(conf), 4),
            "model_version": used_version,
        })

    # Bulk insert all logs into PostgreSQL in ONE single transaction
    db.add_all(logs)
    await db.commit()

    diabetic_count = sum(1 for r in results if r["predicted_label"] == 1)

    return {
        "batch_id": batch_id,
        "count": len(results),
        "total_rows_in_file": total_in_file,
        "note": f"Processed first {len(results)} rows" if total_in_file > MAX_BATCH_ROWS else "Processed all rows",
        "summary": {
            "diabetic": diabetic_count,
            "non_diabetic": len(results) - diabetic_count,
        },
        "results": results[:100],
    }


# ── Model Version Endpoints ───────────────────────────────────────────────────
@app.get("/models", tags=["Models"])
async def list_models(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ModelVersion).order_by(ModelVersion.version))
    models = result.scalars().all()
    return [
        {
            "version": m.version,
            "model_type": m.model_type,
            "is_active": (m.version == registry.active_version),
            "created_at": format_iso(m.created_at),
        }
        for m in models
    ]


@app.post("/models/{version}/activate", tags=["Models"])
@app.put("/models/{version}/activate", tags=["Models"])
async def activate_model(version: str, db: AsyncSession = Depends(get_db)):
    try:
        registry.set_active_version(version)
    except ModelLoadError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    await db.execute(update(ModelVersion).values(is_active=False))
    await db.execute(update(ModelVersion).where(ModelVersion.version == version).values(is_active=True))
    await db.commit()

    return {"version": version, "is_active": True}


# ── Predictions History & Batches ──────────────────────────────────────────────
@app.get("/batches", tags=["History"])
async def list_batches(limit: int = 20, db: AsyncSession = Depends(get_db)):
    stmt = (
        select(
            PredictionLog.batch_id,
            PredictionLog.model_version,
            func.min(PredictionLog.created_at).label("created_at"),
            func.count(PredictionLog.id).label("count"),
            func.sum(case((PredictionLog.predicted_label == 1, 1), else_=0)).label("diabetic_count"),
        )
        .where(PredictionLog.batch_id.isnot(None))
        .group_by(PredictionLog.batch_id, PredictionLog.model_version)
        .order_by(desc(func.min(PredictionLog.created_at)))
        .limit(limit)
    )
    res = await db.execute(stmt)
    rows = res.all()
    return [
        {
            "batch_id": r.batch_id,
            "model_version": r.model_version,
            "created_at": format_iso(r.created_at),
            "count": r.count,
            "diabetic": int(r.diabetic_count or 0),
            "non_diabetic": int(r.count - (r.diabetic_count or 0)),
        }
        for r in rows
    ]


@app.get("/batches/{batch_id}", tags=["History"])
async def get_batch_results(batch_id: str, limit: int = 100, db: AsyncSession = Depends(get_db)):
    stmt = (
        select(PredictionLog)
        .where(PredictionLog.batch_id == batch_id)
        .order_by(PredictionLog.created_at.asc())
        .limit(limit)
    )
    res = await db.execute(stmt)
    logs = res.scalars().all()
    if not logs:
        raise HTTPException(status_code=404, detail="Batch not found")
    diabetic = sum(1 for l in logs if l.predicted_label == 1)
    return {
        "batch_id": batch_id,
        "model_version": logs[0].model_version if logs else None,
        "count": len(logs),
        "created_at": format_iso(logs[0].created_at) if logs else None,
        "summary": {
            "diabetic": diabetic,
            "non_diabetic": len(logs) - diabetic,
        },
        "results": [
            {
                "id": l.id,
                "row": idx + 1,
                "predicted_label": l.predicted_label,
                "label": "Diabetes" if l.predicted_label == 1 else "No Diabetes",
                "label_name": LABEL_NAMES.get(l.predicted_label, "unknown"),
                "confidence": l.confidence,
                "model_version": l.model_version,
                "latency_ms": l.latency_ms,
            }
            for idx, l in enumerate(logs)
        ]
    }


@app.get("/predictions", tags=["History"])
@app.get("/inferences", tags=["History"])
async def list_predictions(
    limit: int = 50,
    kind: Optional[str] = None,
    batch_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(PredictionLog)
    if kind == "single":
        query = query.where(PredictionLog.batch_id.is_(None))
    elif kind == "batch":
        query = query.where(PredictionLog.batch_id.isnot(None))
    if batch_id:
        query = query.where(PredictionLog.batch_id == batch_id)

    query = query.order_by(PredictionLog.created_at.desc()).limit(limit)
    result = await db.execute(query)
    logs = result.scalars().all()
    return [
        {
            "id": l.id,
            "batch_id": l.batch_id,
            "model_version": l.model_version,
            "predicted_label": l.predicted_label,
            "label_name": LABEL_NAMES.get(l.predicted_label, "unknown"),
            "confidence": l.confidence,
            "status": l.status,
            "created_at": format_iso(l.created_at),
        }
        for l in logs
    ]


@app.get("/predictions/{prediction_id}", tags=["History"])
async def get_prediction(prediction_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(PredictionLog).where(PredictionLog.id == prediction_id)
    )
    log = result.scalar_one_or_none()
    if not log:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return {
        "id": log.id,
        "batch_id": log.batch_id,
        "model_version": log.model_version,
        "input_payload": log.input_payload,
        "input_hash": log.input_hash,
        "predicted_label": log.predicted_label,
        "label_name": LABEL_NAMES.get(log.predicted_label, "unknown"),
        "confidence": log.confidence,
        "status": log.status,
        "latency_ms": log.latency_ms,
        "created_at": format_iso(log.created_at),
    }
