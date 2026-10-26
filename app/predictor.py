"""Inference: ONNX Runtime-ზე, PyTorch-ის დამოკიდებულების გარეშე.

სერვისს torch არ სჭირდება — მხოლოდ onnxruntime, Pillow და numpy.
ეს ასობით მეგაბაიტით ამცირებს deployment-ის ზომას.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
MODEL_PATH = ROOT / "models" / "mobilenet_pet.onnx"
CLASSES_PATH = APP_DIR / "classes.json"
META_PATH = APP_DIR / "model_meta.json"


@lru_cache(maxsize=1)
def load_runtime():
    """მოდელი ერთხელ იტვირთება და ყველა მოთხოვნაზე ხელახლა იყენებს."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"მოდელი ვერ მოიძებნა: {MODEL_PATH}. "
            "გაუშვი `python -m src.export_onnx` ან ჩადე გაწვრთნილი .onnx ფაილი models/-ში."
        )
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    mean = np.array(meta["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.array(meta["std"], dtype=np.float32).reshape(3, 1, 1)
    size = tuple(meta["input_size"])
    return session, classes, mean, std, size


def preprocess(image: Image.Image) -> np.ndarray:
    """ზუსტად იმეორებს ვარჯიშის eval_tf-ს: Resize -> ToTensor -> Normalize.

    თუ აქ სხვა დამუშავებას გამოვიყენებთ, მოდელი სხვა განაწილებას მიიღებს
    და პროგნოზები გაუარესდება — ეს inference-ის ყველაზე ხშირი შეცდომაა.
    """
    _, _, mean, std, size = load_runtime()
    image = image.convert("RGB").resize(size)
    arr = np.asarray(image, dtype=np.float32) / 255.0   # HWC, 0..1
    arr = arr.transpose(2, 0, 1)                        # CHW
    arr = (arr - mean) / std
    return arr[None, ...].astype(np.float32)            # NCHW


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def predict(image: Image.Image, top_k: int = 3):
    """აბრუნებს top-k ჯიშს confidence-ით, ყველაზე სავარაუდოდან დაწყებული."""
    session, classes, *_ = load_runtime()
    logits = session.run(None, {"image": preprocess(image)})[0][0]
    probs = softmax(logits)
    top_k = max(1, min(top_k, len(classes)))
    order = np.argsort(probs)[::-1][:top_k]
    return [{"breed": classes[i], "confidence": round(float(probs[i]), 4)} for i in order]


def class_names():
    return load_runtime()[1]
