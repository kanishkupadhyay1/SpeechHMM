"""
FastAPI Application for Speech Command Recognition using Hidden Markov Models.
"""

import sys
import os

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.schemas import (
    HealthResponse,
    CommandsResponse,
    MetricSummary,
    ConfusionMatrixResponse,
    ModelAnalysisResponse,
    RecognitionResponse,
    MFCCDataResponse
)
from src.api.prediction_service import PredictionService

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("speech_hmm_api")

app = FastAPI(
    title="Speech Command Recognition HMM API",
    description="Production-grade acoustic command recognition using Hidden Markov Models trained on Google Speech Commands v0.02.",
    version="1.0.0"
)

# CORS setup for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction service
service = PredictionService()

# Serve static plots if available
plots_dir = os.path.join(PROJECT_ROOT, "results", "plots")
if os.path.exists(plots_dir):
    app.mount("/plots", StaticFiles(directory=plots_dir), name="plots")

# Serve frontend build assets if available
dist_dir = os.path.join(PROJECT_ROOT, "frontend", "dist")
dist_assets = os.path.join(dist_dir, "assets")
if os.path.exists(dist_assets):
    app.mount("/assets", StaticFiles(directory=dist_assets), name="assets")


@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Speech Command Recognition HMM API...")
    service.load_models()
    logger.info(f"Loaded commands: {service.commands}")


@app.get("/")
def root_endpoint(request: Request):
    """Health check endpoint or frontend index depending on accept header."""
    accept = request.headers.get("accept", "")
    index_path = os.path.join(dist_dir, "index.html")
    if "text/html" in accept and os.path.exists(index_path):
        return FileResponse(index_path)

    return {
        "status": "healthy" if service.is_loaded else "ready_for_training",
        "message": "AI-Powered Speech Command Recognition API is operational.",
        "models_loaded": len(service.models),
        "commands": service.commands
    }


@app.get("/api/commands", response_model=CommandsResponse)
def get_commands():
    """List supported vocabulary commands."""
    return {
        "commands": service.commands,
        "count": len(service.commands)
    }


@app.get("/api/metrics")
def get_metrics():
    """Fetch official test set evaluation metrics."""
    metrics_path = os.path.join("results", "metrics", "evaluation_metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=404,
            detail="Evaluation metrics not found. Please run model evaluation first."
        )
    with open(metrics_path, "r") as f:
        data = json.load(f)
    return data


@app.get("/api/confusion-matrix", response_model=ConfusionMatrixResponse)
def get_confusion_matrix():
    """Fetch confusion matrix from actual held-out test predictions."""
    metrics_path = os.path.join("results", "metrics", "evaluation_metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=404,
            detail="Confusion matrix not found. Please run model evaluation first."
        )
    with open(metrics_path, "r") as f:
        data = json.load(f)
    return {
        "commands": data["commands"],
        "matrix": data["confusion_matrix"],
        "accuracy": data["accuracy"],
        "macro_f1": data["macro_f1"]
    }


@app.get("/api/model-analysis")
def get_model_analysis():
    """Detailed architectural and parameter inspection of trained HMMs."""
    details = {}
    for cmd, model in service.models.items():
        details[cmd] = {
            "n_components": model.n_components,
            "covariance_type": model.covariance_type,
            "converged": bool(getattr(model.monitor_, "converged", True)),
            "iterations": int(getattr(model.monitor_, "iter", 0)),
            "transition_matrix_shape": list(model.transmat_.shape),
            "means_shape": list(model.means_.shape)
        }

    metrics = None
    metrics_path = os.path.join("results", "metrics", "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    return {
        "commands": service.commands,
        "n_states": service.config.get("n_states", 5),
        "covariance_type": service.config.get("covariance_type", "diag"),
        "feature_dimensions": service.config.get("extractor_config", {}).get("feature_dim", 39),
        "metrics": metrics,
        "model_details": details
    }


@app.post("/api/recognize", response_model=RecognitionResponse)
async def recognize_audio(file: UploadFile = File(...)):
    """
    Accept uploaded WAV audio, preprocess, compute MFCCs, score all command HMMs,
    and return predicted command with log likelihood scores.
    """
    if not service.is_loaded:
        service.load_models()
        if not service.is_loaded:
            raise HTTPException(status_code=503, detail="HMM models not loaded. Please train models first.")

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file submitted.")
        result = service.recognize_audio(content)
        return result
    except Exception as e:
        logger.error(f"Recognition error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audio recognition failed: {str(e)}")


@app.post("/api/recognize-all")
async def recognize_all_likelihoods(file: UploadFile = File(...)):
    """Return raw log-likelihood scores and normalized probabilities for all commands."""
    return await recognize_audio(file)


@app.post("/api/mfcc", response_model=MFCCDataResponse)
async def extract_mfcc_visualization(file: UploadFile = File(...)):
    """Extract MFCC coefficient matrix for visualization in charts."""
    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file submitted.")
        data = service.compute_mfcc_data(content)
        return data
    except Exception as e:
        logger.error(f"MFCC extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"MFCC extraction failed: {str(e)}")


@app.get("/{full_path:path}")
async def serve_spa_fallback(full_path: str):
    """Fallback handler to serve React SPA routes when deployed."""
    if full_path.startswith("api") or full_path.startswith("plots") or full_path.startswith("docs") or full_path.startswith("openapi.json"):
        raise HTTPException(status_code=404, detail="Endpoint not found")

    file_path = os.path.join(dist_dir, full_path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)

    index_path = os.path.join(dist_dir, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)

    raise HTTPException(status_code=404, detail="Page not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
