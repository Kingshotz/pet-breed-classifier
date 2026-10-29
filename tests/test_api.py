"""API-ის მინიმალური ტესტები: სერვისი იწევა და არასწორ შეყვანას სწორად ამუშავებს."""
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "mobilenet_pet.onnx"


def make_image_bytes(size=(224, 224), color=(120, 90, 60)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health():
    response = client.get("/health")
    assert response.status_code in (200, 503)   # 503, თუ მოდელის ფაილი ჯერ არ არის
    if response.status_code == 200:
        assert response.json()["num_classes"] == 37


def test_predict_rejects_non_image():
    response = client.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_predict_rejects_corrupt_image():
    response = client.post("/predict", files={"file": ("a.jpg", b"not-an-image", "image/jpeg")})
    assert response.status_code == 400


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="მოდელის ფაილი არ არის")
def test_predict_returns_top_k():
    response = client.post("/predict",
                           files={"file": ("pet.jpg", make_image_bytes(), "image/jpeg")},
                           params={"top_k": 3})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 3
    assert predictions[0]["confidence"] >= predictions[-1]["confidence"]
    assert all(0.0 <= p["confidence"] <= 1.0 for p in predictions)
