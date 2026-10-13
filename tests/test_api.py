"""
tests/test_api.py — PyTest Suite for DeepFake Detection API
Tests cover:
- Health checks
- Single image inference & schema validation
- Probabilities summation & bounds
- Non-image rejection
- Grad-CAM explanation endpoint
- Video inference with synthetic sample
"""

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app import app

client = TestClient(app)


def make_test_image(width=224, height=224, color=None):
    """Generate a test image buffer."""
    if color:
        img = Image.new("RGB", (width, height), color=color)
    else:
        arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_health_check():
    """Verify health endpoint reports healthy status."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_root_endpoint():
    """Verify root endpoint returns system metadata."""
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert "version" in data
    assert "architecture" in data


def test_predict_returns_valid_json():
    """Verify prediction returns valid structure and labels."""
    img_buf = make_test_image()
    res = client.post(
        "/predict",
        files={"file": ("test.jpg", img_buf, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["prediction"] in ["Real", "Fake"]
    assert 0 <= data["confidence"] <= 100.0
    assert "real_probability" in data
    assert "fake_probability" in data


def test_predict_returns_model_breakdown():
    """Verify prediction includes breakdown of architecture components."""
    img_buf = make_test_image()
    res = client.post(
        "/predict",
        files={"file": ("face.jpg", img_buf, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert "model_breakdown" in data
    breakdown = data["model_breakdown"]
    assert "resnext_spatial" in breakdown
    assert "lstm_temporal" in breakdown


def test_reject_non_image():
    """Verify 400 Bad Request when uploading non-image file."""
    fake_txt = io.BytesIO(b"This is not an image.")
    res = client.post(
        "/predict",
        files={"file": ("sample.txt", fake_txt, "text/plain")}
    )
    assert res.status_code == 400


def test_probabilities_sum_to_100():
    """Verify real + fake probabilities sum to ~100%."""
    img_buf = make_test_image()
    res = client.post(
        "/predict",
        files={"file": ("check.jpg", img_buf, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    total = data["real_probability"] + data["fake_probability"]
    assert abs(total - 100.0) < 0.2


def test_predict_explain_returns_gradcam():
    """Verify Grad-CAM explanation returns base64 heatmap image."""
    img_buf = make_test_image()
    res = client.post(
        "/predict/explain",
        files={"file": ("explain.jpg", img_buf, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "heatmap_base64" in data
    assert len(data["heatmap_base64"]) > 50
    assert "heatmap_note" in data


def test_batch_prediction():
    """Verify batch endpoint handles multiple images."""
    img1 = make_test_image()
    img2 = make_test_image()
    res = client.post(
        "/predict/batch",
        files=[
            ("files", ("img1.jpg", img1, "image/jpeg")),
            ("files", ("img2.jpg", img2, "image/jpeg")),
        ]
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["total"] == 2
    assert len(data["results"]) == 2


def test_reject_invalid_video_format():
    """Verify rejection of unsupported video format."""
    bad_file = io.BytesIO(b"fake data")
    res = client.post(
        "/predict/video",
        files={"file": ("video.txt", bad_file, "text/plain")}
    )
    assert res.status_code == 400


def test_video_prediction_synthetic():
    """Verify video analysis on synthetic video stream."""
    import os
    import tempfile

    import cv2

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_path = tmp.name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_path, fourcc, 5.0, (64, 64))
        for _ in range(10):
            frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

        with open(temp_path, "rb") as vf:
            res = client.post(
                "/predict/video?sample_fps=2&max_frames=10",
                files={"file": ("sample.mp4", vf, "video/mp4")}
            )

        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["overall_verdict"] in ["Real", "Fake"]
        assert "timeline" in data
        assert len(data["timeline"]) > 0
        assert "suspicious_segments" in data
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

