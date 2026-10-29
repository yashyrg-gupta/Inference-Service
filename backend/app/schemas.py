from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from uuid import UUID
from datetime import datetime
import hashlib
import json


# ─────────────────────────────────────────────────────────────────────────────
# Input Schema — strict validation for all 21 CDC health indicator features
# ─────────────────────────────────────────────────────────────────────────────

class PatientRecord(BaseModel):
    """
    Single patient health indicator record.
    All fields match the CDC BRFSS diabetes dataset exactly.
    Binary fields must be 0 or 1. Range fields validated strictly.
    """

    HighBP: int = Field(..., ge=0, le=1, description="High blood pressure (0=No, 1=Yes)")
    HighChol: int = Field(..., ge=0, le=1, description="High cholesterol (0=No, 1=Yes)")
    CholCheck: int = Field(..., ge=0, le=1, description="Cholesterol check in last 5 years (0=No, 1=Yes)")
    BMI: float = Field(..., ge=10.0, le=100.0, description="Body Mass Index (10.0 - 100.0)")
    Smoker: int = Field(..., ge=0, le=1, description="Smoked at least 100 cigarettes in lifetime (0=No, 1=Yes)")
    Stroke: int = Field(..., ge=0, le=1, description="Ever had a stroke (0=No, 1=Yes)")
    HeartDiseaseorAttack: int = Field(..., ge=0, le=1, description="Coronary heart disease or heart attack (0=No, 1=Yes)")
    PhysActivity: int = Field(..., ge=0, le=1, description="Physical activity in past 30 days (0=No, 1=Yes)")
    Fruits: int = Field(..., ge=0, le=1, description="Consume fruit 1+ times per day (0=No, 1=Yes)")
    Veggies: int = Field(..., ge=0, le=1, description="Consume vegetables 1+ times per day (0=No, 1=Yes)")
    HvyAlcoholConsump: int = Field(..., ge=0, le=1, description="Heavy alcohol consumption (0=No, 1=Yes)")
    AnyHealthcare: int = Field(..., ge=0, le=1, description="Any healthcare coverage (0=No, 1=Yes)")
    NoDocbcCost: int = Field(..., ge=0, le=1, description="Could not see doctor due to cost (0=No, 1=Yes)")
    GenHlth: int = Field(..., ge=1, le=5, description="General health rating (1=Excellent, 5=Poor)")
    MentHlth: int = Field(..., ge=0, le=30, description="Days mental health not good (0-30)")
    PhysHlth: int = Field(..., ge=0, le=30, description="Days physical health not good (0-30)")
    DiffWalk: int = Field(..., ge=0, le=1, description="Difficulty walking/climbing stairs (0=No, 1=Yes)")
    Sex: int = Field(..., ge=0, le=1, description="Sex (0=Female, 1=Male)")
    Age: int = Field(..., ge=1, le=13, description="Age category (1=18-24, 13=80+)")
    Education: int = Field(..., ge=1, le=6, description="Education level (1=None, 6=College grad)")
    Income: int = Field(..., ge=1, le=8, description="Income level (1=<$10k, 8=$75k+)")

    @field_validator("BMI", mode="before")
    @classmethod
    def validate_bmi(cls, v):
        try:
            val = float(v) 
        except (TypeError, ValueError):
            raise ValueError("BMI must be a number")
        return val

    def to_feature_list(self) -> list:
        """Return features in the exact order the model expects."""
        return [
            self.HighBP, self.HighChol, self.CholCheck, self.BMI,
            self.Smoker, self.Stroke, self.HeartDiseaseorAttack,
            self.PhysActivity, self.Fruits, self.Veggies,
            self.HvyAlcoholConsump, self.AnyHealthcare, self.NoDocbcCost,
            self.GenHlth, self.MentHlth, self.PhysHlth, self.DiffWalk,
            self.Sex, self.Age, self.Education, self.Income,
        ]

    def input_hash(self) -> str:
        """SHA-256 hash of input for reproducibility."""
        payload = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()

    model_config = {"extra": "forbid"}  # No silent coercion of extra fields


# ─────────────────────────────────────────────────────────────────────────────
# Request Models
# ─────────────────────────────────────────────────────────────────────────────

class SinglePredictRequest(BaseModel):
    record: PatientRecord
    model_version: Optional[str] = Field(
        default=None,
        description="Model version to use ('v1' or 'v2'). Defaults to active version."
    )

    @field_validator("model_version")
    @classmethod
    def validate_model_version(cls, v):
        if v is not None and v not in ("v1", "v2"):
            raise ValueError("model_version must be 'v1' or 'v2'")
        return v


class BatchPredictRequest(BaseModel):
    records: List[PatientRecord] = Field(..., min_length=1, max_length=10000)
    model_version: Optional[str] = Field(default=None)

    @field_validator("model_version")
    @classmethod
    def validate_model_version(cls, v):
        if v is not None and v not in ("v1", "v2"):
            raise ValueError("model_version must be 'v1' or 'v2'")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Response Models
# ─────────────────────────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    prediction_id: UUID
    label: str
    label_int: int
    confidence: float
    model_version: str
    model_name: str
    input_hash: str
    preprocessing_version: str
    timestamp: datetime
    latency_ms: int
    input_echo: dict


class BatchPredictionResult(BaseModel):
    batch_id: UUID
    total_records: int
    model_version: str
    results: List[PredictionResult]
    created_at: datetime


class ModelInfo(BaseModel):
    version: str
    is_active: bool
    registered_at: datetime
    file_path: str
    metadata: Optional[dict] = None


class InferenceLogResponse(BaseModel):
    id: UUID
    batch_id: Optional[UUID]
    label: str
    label_int: int
    confidence: float
    model_version: str
    input_hash: str
    latency_ms: int
    created_at: datetime
    input_data: dict


class PaginatedInferenceResponse(BaseModel):
    total: int
    page: int
    page_size: int
    results: List[InferenceLogResponse]


class HealthResponse(BaseModel):
    status: str
    active_model_version: str
    models_loaded: List[str]
    database: str
