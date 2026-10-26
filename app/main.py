"""FastAPI სერვისი ჯიშის კლასიფიკაციისთვის.

გაშვება:
    uvicorn app.main:app --reload
ინტერაქტიული დოკუმენტაცია: http://127.0.0.1:8000/docs
"""
import io

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from app import predictor

app = FastAPI(
    title="Pet Breed Classifier",
    description="Oxford-IIIT Pet, 37 ჯიში. MobileNetV2 fine-tuning, ONNX Runtime (CPU).",
    version="1.0.0",
)

MAX_BYTES = 10 * 1024 * 1024   # 10 MB


@app.get("/health", summary="სერვისისა და მოდელის მდგომარეობა")
def health():
    try:
        classes = predictor.class_names()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"status": "ok", "num_classes": len(classes)}


@app.get("/classes", summary="ჯიშების სრული სია")
def classes():
    return {"classes": predictor.class_names()}


@app.post("/predict", summary="სურათის კლასიფიკაცია")
async def predict_endpoint(
    file: UploadFile = File(..., description="ძაღლის ან კატის სურათი"),
    top_k: int = Query(3, ge=1, le=10),
):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="ატვირთეთ სურათის ფაილი (image/*)")

    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="ფაილი 10 MB-ზე დიდია")

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="ფაილი ვერ წაიკითხა როგორც სურათი")

    try:
        predictions = predictor.predict(image, top_k=top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return {"filename": file.filename, "predictions": predictions}
