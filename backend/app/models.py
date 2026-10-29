import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, JSON
from app.db import Base


class ModelVersion(Base):
    """Registry of available model versions."""
    __tablename__ = "model_versions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    version = Column(String(50), unique=True, nullable=False, index=True)
    model_type = Column(String(100), default="classifier")
    artifact_path = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class PredictionLog(Base):
    """Audit log for each prediction."""
    __tablename__ = "prediction_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    batch_id = Column(String(36), nullable=True, index=True)
    model_version = Column(String(50), nullable=False, index=True)
    input_payload = Column(JSON, nullable=False)
    input_hash = Column(String(64), nullable=False, index=True)
    preprocessing_version = Column(String(20), default="1.0", nullable=False)
    predicted_label = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    latency_ms = Column(Integer, default=0, nullable=False)
    status = Column(String(20), default="success", nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
