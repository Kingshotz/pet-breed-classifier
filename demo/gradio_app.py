"""Gradio დემო — სურათს ტვირთავ და top-3 პროგნოზს იღებ.

გაშვება:
    python demo/gradio_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr

from app import predictor


def classify(image):
    if image is None:
        return {}
    return {p["breed"]: p["confidence"] for p in predictor.predict(image, top_k=3)}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="ატვირთე ძაღლის ან კატის სურათი"),
    outputs=gr.Label(num_top_classes=3, label="პროგნოზი"),
    title="Pet Breed Classifier — 37 ჯიში",
    description=("MobileNetV2 fine-tuning Oxford-IIIT Pet-ზე. "
                 "Test accuracy 88.55%, macro F1 0.884. Inference: ONNX Runtime, CPU."),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
