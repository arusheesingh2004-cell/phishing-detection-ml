"""
app.py

FastAPI backend for the Lightweight AI Framework for Real-Time Phishing
Detection.

Endpoints
---------
GET  /api/health           -> service + model status
GET  /api/model-info       -> metrics captured during training (both models)
POST /api/predict          -> classify a single URL as phishing/legitimate
POST /api/predict/batch    -> classify up to MAX_BATCH_SIZE URLs (JSON list) in one call
POST /api/predict/batch-file -> classify URLs from an uploaded .csv/.txt file
GET  /api/history          -> most recent predictions (from the database)

Database
--------
Uses SQLAlchemy. Defaults to a local SQLite file (zero setup, matches the
"lightweight" theme) but honours DATABASE_URL if set, e.g.:
    postgresql+psycopg2://user:password@localhost:5432/phishing_db
so the exact PostgreSQL setup proposed in the synopsis is a one-line swap.
"""

import os
import io
import csv
import time
import datetime as dt
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

from feature_extractor import (
    extract_features, features_to_vector, FEATURE_NAMES, extract_features_matrix,
)

MAX_BATCH_SIZE = 5000  # sane request-size cap; raise if you need more per call

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./phishing_detector.db")
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, nullable=False)
    prediction = Column(String, nullable=False)      # "phishing" | "legitimate"
    confidence = Column(Float, nullable=False)
    model_used = Column(String, nullable=False)
    latency_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


Base.metadata.create_all(bind=engine)

# ---------------------------------------------------------------------------
# Load trained model artifacts
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

_best_model_name = "xgboost"
try:
    with open(os.path.join(MODEL_DIR, "best_model.txt")) as f:
        _best_model_name = f.read().strip()
except FileNotFoundError:
    pass

_xgb_model = None
_logreg_model = None
_scaler = None
_metrics = {}

try:
    _xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost.joblib"))
    _logreg_model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.joblib"))
    _scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    import json
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        _metrics = json.load(f)
except FileNotFoundError:
    pass  # app.py can still start; /predict will raise a clear error


def _get_model(name: str):
    if name == "logistic_regression":
        return _logreg_model, True
    return _xgb_model, False


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Lightweight AI Phishing Detection API",
    description="Real-time phishing URL classification using lightweight lexical features.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # relax for local student-project use; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048, example="http://appleid.apple.com-app.es/verify")
    model: str = Field(
        default="xgboost",
        description="Which trained model to use: 'xgboost' or 'logistic_regression'.",
    )


class PredictResponse(BaseModel):
    url: str
    prediction: str
    confidence: float
    model_used: str
    latency_ms: float
    top_signals: list


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models_loaded": _xgb_model is not None and _logreg_model is not None,
        "best_model": _best_model_name,
    }


@app.get("/api/model-info")
def model_info():
    if not _metrics:
        raise HTTPException(status_code=503, detail="Model metrics not available. Run train_model.py first.")
    return _metrics


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _xgb_model is None or _logreg_model is None:
        raise HTTPException(status_code=503, detail="Models not trained yet. Run train_model.py first.")

    model_choice = req.model if req.model in ("xgboost", "logistic_regression") else "xgboost"
    model, needs_scaling = _get_model(model_choice)

    t0 = time.perf_counter()
    feats = extract_features(req.url)
    vec = np.array([features_to_vector(feats)], dtype=float)
    if needs_scaling:
        vec = _scaler.transform(vec)

    proba = model.predict_proba(vec)[0]
    phishing_proba = float(proba[1])
    label = "phishing" if phishing_proba >= 0.5 else "legitimate"
    confidence = phishing_proba if label == "phishing" else 1 - phishing_proba
    latency_ms = (time.perf_counter() - t0) * 1000

    # Surface the strongest contributing signals for this specific URL
    # (using global XGBoost feature importances, filtered to features that
    # actually fired for this URL, as a lightweight explainability aid).
    importances = _metrics.get("feature_importance_xgboost", {})
    fired = [name for name in FEATURE_NAMES if feats.get(name, 0) not in (0, 0.0)]
    top_signals = sorted(fired, key=lambda n: -importances.get(n, 0))[:5]
    top_signals = [{"feature": n, "value": feats[n]} for n in top_signals]

    # Log to DB (best-effort; don't fail the request if logging fails)
    try:
        db = SessionLocal()
        db.add(PredictionLog(
            url=req.url,
            prediction=label,
            confidence=round(confidence, 4),
            model_used=model_choice,
            latency_ms=round(latency_ms, 4),
        ))
        db.commit()
        db.close()
    except Exception:
        pass

    return PredictResponse(
        url=req.url,
        prediction=label,
        confidence=round(confidence, 4),
        model_used=model_choice,
        latency_ms=round(latency_ms, 4),
        top_signals=top_signals,
    )


class BatchPredictRequest(BaseModel):
    urls: List[str] = Field(..., min_items=1, max_items=MAX_BATCH_SIZE)
    model: str = Field(default="xgboost")


def _score_urls(urls: List[str], model_choice: str, n_jobs: int = 1):
    """
    Shared scoring core for both batch endpoints: one vectorized feature
    matrix + one predict_proba call for the whole list, instead of looping
    model.predict_proba per URL. This is what makes scoring a few thousand
    URLs fast -- the cost is dominated by feature extraction, not by the
    number of separate model calls.
    """
    model, needs_scaling = _get_model(model_choice)
    if model is None:
        raise HTTPException(status_code=503, detail="Models not trained yet. Run train_model.py first.")

    t0 = time.perf_counter()
    X = extract_features_matrix(urls, n_jobs=n_jobs)
    if needs_scaling:
        X = _scaler.transform(X)
    proba = model.predict_proba(X)[:, 1]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    results = []
    for url, p in zip(urls, proba):
        p = float(p)
        label = "phishing" if p >= 0.5 else "legitimate"
        confidence = p if label == "phishing" else 1 - p
        results.append({"url": url, "prediction": label, "confidence": round(confidence, 4)})

    # Bulk log to DB in one transaction (best-effort)
    try:
        db = SessionLocal()
        db.add_all([
            PredictionLog(
                url=r["url"], prediction=r["prediction"], confidence=r["confidence"],
                model_used=model_choice, latency_ms=round(elapsed_ms / len(urls), 4),
            )
            for r in results
        ])
        db.commit()
        db.close()
    except Exception:
        pass

    return results, elapsed_ms


@app.post("/api/predict/batch")
def predict_batch(req: BatchPredictRequest):
    """Score up to MAX_BATCH_SIZE URLs in one request -- for scoring lists/logs, not just single lookups."""
    model_choice = req.model if req.model in ("xgboost", "logistic_regression") else "xgboost"
    results, elapsed_ms = _score_urls(req.urls, model_choice, n_jobs=1)
    phishing_count = sum(1 for r in results if r["prediction"] == "phishing")
    return {
        "count": len(results),
        "phishing_count": phishing_count,
        "legitimate_count": len(results) - phishing_count,
        "total_latency_ms": round(elapsed_ms, 2),
        "avg_latency_ms_per_url": round(elapsed_ms / len(results), 4),
        "results": results,
    }


@app.post("/api/predict/batch-file")
async def predict_batch_file(file: UploadFile = File(...), model: str = "xgboost"):
    """
    Score URLs from an uploaded .csv or .txt file. CSV: uses a 'url' column
    if present, otherwise the first column. TXT: one URL per line.
    """
    raw = (await file.read()).decode("utf-8", errors="ignore")
    urls: List[str] = []

    if file.filename and file.filename.lower().endswith(".csv"):
        rows = list(csv.reader(io.StringIO(raw)))
        if not rows:
            raise HTTPException(status_code=400, detail="Empty file.")
        header = [h.strip().lower() for h in rows[0]]
        has_named_url_col = "url" in header
        url_idx = header.index("url") if has_named_url_col else 0
        first_cell_is_url = bool(rows[0]) and rows[0][0].strip().startswith(("http://", "https://"))
        # Skip row 0 as a header unless it's clearly already data (starts with a URL).
        start = 0 if first_cell_is_url else 1
        for row in rows[start:]:
            if row and len(row) > url_idx and row[url_idx].strip():
                urls.append(row[url_idx].strip())
    else:
        urls = [line.strip() for line in raw.splitlines() if line.strip()]

    if not urls:
        raise HTTPException(status_code=400, detail="No URLs found in the uploaded file.")
    if len(urls) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"File has {len(urls)} URLs; max per request is {MAX_BATCH_SIZE}.")

    model_choice = model if model in ("xgboost", "logistic_regression") else "xgboost"
    results, elapsed_ms = _score_urls(urls, model_choice, n_jobs=1)
    phishing_count = sum(1 for r in results if r["prediction"] == "phishing")
    return {
        "filename": file.filename,
        "count": len(results),
        "phishing_count": phishing_count,
        "legitimate_count": len(results) - phishing_count,
        "total_latency_ms": round(elapsed_ms, 2),
        "avg_latency_ms_per_url": round(elapsed_ms / len(results), 4),
        "results": results,
    }


@app.get("/api/history")
def history(limit: int = 20):
    db = SessionLocal()
    rows = (
        db.query(PredictionLog)
        .order_by(PredictionLog.id.desc())
        .limit(min(limit, 100))
        .all()
    )
    db.close()
    return [
        {
            "url": r.url,
            "prediction": r.prediction,
            "confidence": r.confidence,
            "model_used": r.model_used,
            "latency_ms": r.latency_ms,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Serve the frontend (mounted LAST so it never shadows the /api/* routes above)
# ---------------------------------------------------------------------------
_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
