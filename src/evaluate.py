"""საბოლოო შეფასება სატესტო ნაკრებზე + შეცდომების ანალიზი.

გაშვება:
    python -m src.evaluate --checkpoint models/E_mobilenet_v2_best.pth --arch mobilenet
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from src import config, models
from src.data import build_dataloaders
from src.train import get_device


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    preds, targets = [], []
    for images, labels in loader:
        outputs = model(images.to(device))
        preds.append(outputs.argmax(1).cpu())
        targets.append(labels)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def top_confusions(cm, class_names, k=10):
    """ყველაზე ხშირი აღრევები დიაგონალის გარეშე."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = [(int(off[i, j]), class_names[i], class_names[j])
             for i in range(len(class_names)) for j in range(len(class_names))
             if off[i, j] > 0]
    pairs.sort(reverse=True)
    return pairs[:k]


def save_confusion_matrix(cm, path, title="Confusion matrix"):
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.colorbar()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(config.MODELS_DIR / "E_mobilenet_v2_best.pth"))
    parser.add_argument("--arch", default="mobilenet", choices=["mobilenet", "resnet18", "baseline"])
    args = parser.parse_args()

    device = get_device()
    _, _, test_loader, class_names = build_dataloaders()

    if args.arch == "mobilenet":
        model = models.build_mobilenet_v2(pretrained=False)
    elif args.arch == "resnet18":
        model = models.build_resnet18(pretrained=False)
    else:
        model = models.SimpleCNN()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    y_pred, y_true = predict_all(model, test_loader, device)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"test accuracy: {acc:.4f}")
    print(f"macro F1:      {macro_f1:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    print("\nmost frequent confusions (true -> predicted):")
    for count, true_c, pred_c in top_confusions(cm, class_names):
        print(f"  {count:2d}x  {true_c} -> {pred_c}")

    per_class = cm.diagonal() / cm.sum(1)
    print("\nhardest classes:")
    for i in np.argsort(per_class)[:5]:
        print(f"  {class_names[i]}: {per_class[i]:.1%}")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(cm, config.REPORTS_DIR / "confusion_matrix.png",
                          f"Confusion matrix ({args.arch}, test)")
    json.dump({"accuracy": float(acc), "macro_f1": float(macro_f1),
               "checkpoint": args.checkpoint},
              open(config.REPORTS_DIR / "test_metrics.json", "w"), indent=2)
    print("\nreports saved to", config.REPORTS_DIR)


if __name__ == "__main__":
    main()
