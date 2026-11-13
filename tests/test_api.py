"""
Tests for FastAPI application endpoints.
"""

import os
import io
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_api_health_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "commands" in data
    assert len(data["commands"]) == 10


def test_api_commands_endpoint():
    response = client.get("/api/commands")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 10
    assert "yes" in data["commands"]
    assert "stop" in data["commands"]


def test_api_mfcc_endpoint_with_real_audio():
    # Use real wav file from dataset
    wav_path = os.path.join("data", "raw", "speech_commands_v0.02", "yes", "004ae714_nohash_0.wav")
    if not os.path.exists(wav_path):
        pytest.skip("Raw WAV not found for API test")

    with open(wav_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/mfcc",
        files={"file": ("test.wav", file_bytes, "audio/wav")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["feature_dim"] == 39
    assert data["time_frames"] > 0
    assert "mfcc_matrix" in data
    assert len(data["mfcc_matrix"]) == data["time_frames"]


def test_api_metrics_endpoint():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "accuracy" in data
    assert "macro_f1" in data
    assert "confusion_matrix" in data
    assert data["num_test_samples"] == 4074


def test_api_confusion_matrix_endpoint():
    response = client.get("/api/confusion-matrix")
    assert response.status_code == 200
    data = response.json()
    assert "matrix" in data
    assert len(data["commands"]) == 10
    assert len(data["matrix"]) == 10


def test_api_recognize_endpoint_with_real_audio():
    wav_path = os.path.join("data", "raw", "speech_commands_v0.02", "yes", "004ae714_nohash_0.wav")
    if not os.path.exists(wav_path):
        pytest.skip("Raw WAV not found for API test")

    with open(wav_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/recognize",
        files={"file": ("yes_sample.wav", file_bytes, "audio/wav")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_command" in data
    assert "scores" in data
    assert len(data["scores"]) == 10
    assert "viterbi_state_path" in data
    assert len(data["viterbi_state_path"]) > 0
