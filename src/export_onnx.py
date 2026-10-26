"""PyTorch -> ONNX ექსპორტი, ერთ ფაილად შეკვრა და CPU ბენჩმარკი.

გაშვება:
    python -m src.export_onnx
"""
import argparse
import json
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src import config, models
from src.data import build_dataloaders


def benchmark_torch(model, x, runs=100):
    with torch.no_grad():
        for _ in range(10):
            model(x)                     # warm-up
        start = time.time()
        for _ in range(runs):
            model(x)
    return (time.time() - start) / runs * 1000


def benchmark_onnx(session, x_np, runs=100):
    for _ in range(10):
        session.run(None, {"image": x_np})
    start = time.time()
    for _ in range(runs):
        session.run(None, {"image": x_np})
    return (time.time() - start) / runs * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(config.MODELS_DIR / "E_mobilenet_v2_best.pth"))
    parser.add_argument("--out", default=str(config.MODELS_DIR / "mobilenet_pet.onnx"))
    args = parser.parse_args()

    model = models.build_mobilenet_v2(pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    tmp_path = args.out.replace(".onnx", "_raw.onnx")
    torch.onnx.export(
        model, dummy, tmp_path,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
    )
    # ახალი ექსპორტერი წონებს ხანდახან ცალკე .data ფაილში ინახავს;
    # ერთ ფაილად ვკრებთ, რომ სერვისს ერთი არტეფაქტის გადატანა ეყოს.
    proto = onnx.load(tmp_path, load_external_data=True)
    onnx.save_model(proto, args.out, save_as_external_data=False)
    for leftover in (tmp_path, tmp_path + ".data"):
        if os.path.exists(leftover):
            os.remove(leftover)

    x = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    x_np = x.numpy()
    session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])

    t_torch = benchmark_torch(model, x)
    t_onnx = benchmark_onnx(session, x_np)
    with torch.no_grad():
        out_torch = model(x).numpy()
    out_onnx = session.run(None, {"image": x_np})[0]
    max_diff = float(np.abs(out_torch - out_onnx).max())

    assert max_diff < 1e-4, f"ONNX output diverged: {max_diff}"

    size_pth = os.path.getsize(args.checkpoint) / 1e6
    size_onnx = os.path.getsize(args.out) / 1e6
    print(f"PyTorch .pth : {size_pth:6.2f} MB | {t_torch:6.1f} ms/image (CPU)")
    print(f"ONNX         : {size_onnx:6.2f} MB | {t_onnx:6.1f} ms/image (CPU)")
    print(f"speed-up     : {t_torch / t_onnx:.2f}x | max output diff: {max_diff:.2e}")

    # API-ს სჭირდება კლასების სია და preprocessing-ის პარამეტრები
    _, _, _, class_names = build_dataloaders(download=True)
    json.dump(list(class_names), open(config.ROOT / "app" / "classes.json", "w"),
              ensure_ascii=False, indent=2)
    json.dump({"model": "mobilenet_v2", "input_size": [config.IMAGE_SIZE, config.IMAGE_SIZE],
               "mean": config.MEAN, "std": config.STD, "num_classes": len(class_names),
               "torch_ms": round(t_torch, 1), "onnx_ms": round(t_onnx, 1)},
              open(config.ROOT / "app" / "model_meta.json", "w"), indent=2)
    print("exported:", args.out)


if __name__ == "__main__":
    main()
