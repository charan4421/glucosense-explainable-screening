"""Async FastAPI service with Pydantic input validation.

v1 PROTOTYPE, not wired to the v2 study: no v2 script writes models/pipeline.joblib. For v2
scoring use scripts/demo.py (README, "Earlier v1 design").

Run: uvicorn src.app.api:app --reload
Expects a fitted pipeline saved with joblib at models/pipeline.joblib (train it with your own script).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/pipeline.joblib")
app = FastAPI(title="GlucoSense-AI")
_pipe = None


class Patient(BaseModel):
    age: float = Field(ge=18, le=120)
    sex: Literal["male", "female"]
    bmi: float = Field(ge=10, le=80)
    waist_cm: float = Field(ge=40, le=200)
    height_cm: float = Field(ge=100, le=230)
    systolic_bp: float = Field(ge=60, le=260)
    diastolic_bp: float = Field(ge=30, le=160)
    family_history_diabetes: Literal["yes", "no"]
    smoker: Literal["yes", "no"]
    race_eth: str


@app.on_event("startup")
async def load() -> None:
    global _pipe
    if MODEL_PATH.exists():
        _pipe = joblib.load(MODEL_PATH)


@app.get("/health")
async def health() -> dict:
    return {"model_loaded": _pipe is not None}


@app.post("/predict")
async def predict(p: Patient) -> dict:
    if _pipe is None:
        raise HTTPException(503, "Model not loaded")
    risk = float(_pipe.predict_proba(pd.DataFrame([p.model_dump()]))[:, 1][0])
    return {"risk": risk, "note": "Screening aid only, not a diagnosis."}
