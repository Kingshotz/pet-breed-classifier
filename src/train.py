"""ვარჯიშის ციკლი და ექსპერიმენტების გაშვება.

გაშვება:
    python -m src.train --experiment mobilenet
    python -m src.train --experiment all
"""
import argparse
import copy
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src import config, models
from src.data import build_dataloaders


def set_seed(seed=config.SEED):
    """ყველა შემთხვევითობის წყარო ერთ seed-ზე, რომ შედეგი განმეორებადი იყოს."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()          # გრადიენტები გროვდება — ყოველ batch-ზე ვასუფთავებთ
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    """model.eval() — Dropout ითიშება, BatchNorm დაგროვულ სტატისტიკას იყენებს."""
    criterion = criterion or nn.CrossEntropyLoss()
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def run_experiment(name, model, optimizer, train_loader, val_loader, device,
                   scheduler=None, epochs=config.EPOCHS, criterion=None):
    """ერთი ექსპერიმენტი; checkpoint ირჩევა val accuracy-ით (early stopping-ის მარტივი ფორმა).

    val loss ყოველთვის ჩვეულებრივი CrossEntropy-ით იზომება, რომ ექსპერიმენტები
    (მაგ. label smoothing-იანი და უმისოდ) შედარებადი დარჩეს.
    """
    criterion = criterion or nn.CrossEntropyLoss()
    eval_criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val, best_state = 0.0, None
    started = time.time()

    for epoch in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, device, eval_criterion)
        if scheduler is not None:
            scheduler.step()
        for key, value in zip(history, [tr_loss, tr_acc, va_loss, va_acc]):
            history[key].append(value)
        if va_acc > best_val:
            best_val, best_state = va_acc, copy.deepcopy(model.state_dict())
        print(f"[{name}] epoch {epoch + 1}/{epochs} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = config.MODELS_DIR / f"{name}_best.pth"
    torch.save(best_state, checkpoint)
    elapsed = time.time() - started
    print(f"[{name}] best val acc: {best_val:.3f} | {elapsed:.0f}s | saved: {checkpoint}\n")
    return {"name": name, "history": history, "best_val_acc": best_val, "seconds": elapsed}


def build_optimizer(model, head_names, weight_decay=0.0, use_adamw=False):
    """Backbone-ს პატარა lr, ახალ თავს დიდი: ნასწავლი ნიშნები ფრთხილად უნდა შეიცვალოს."""
    head, backbone = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (head if name.startswith(head_names) else backbone).append(param)
    groups = [{"params": backbone, "lr": config.LR_BACKBONE},
              {"params": head, "lr": config.LR_HEAD}]
    groups = [g for g in groups if g["params"]]
    cls = optim.AdamW if use_adamw else optim.Adam
    return cls(groups, weight_decay=weight_decay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="mobilenet",
                        choices=["baseline", "frozen", "resnet", "regularized", "mobilenet", "all"])
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    args = parser.parse_args()

    set_seed()
    device = get_device()
    print("device:", device)
    train_loader, val_loader, _, class_names = build_dataloaders()
    print(f"classes: {len(class_names)}")

    results = {}

    def run(key):
        if key == "baseline":
            model = models.SimpleCNN()
            opt = optim.Adam(model.parameters(), lr=config.LR_HEAD)
            return run_experiment("A_baseline_cnn", model, opt, train_loader, val_loader,
                                  device, epochs=max(args.epochs, 15))
        if key == "frozen":
            model = models.build_resnet18(freeze_backbone=True)
            opt = optim.Adam(model.fc.parameters(), lr=config.LR_HEAD)
            return run_experiment("B_resnet18_frozen", model, opt, train_loader, val_loader,
                                  device, epochs=args.epochs)
        if key == "resnet":
            model = models.build_resnet18()
            opt = build_optimizer(model, head_names=("fc.",))
            return run_experiment("C_resnet18_finetune", model, opt, train_loader, val_loader,
                                  device, epochs=args.epochs)
        if key == "regularized":
            model = models.build_resnet18()
            opt = build_optimizer(model, head_names=("fc.",), weight_decay=1e-2, use_adamw=True)
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
            return run_experiment("D_resnet18_regularized", model, opt, train_loader, val_loader,
                                  device, scheduler=sched, epochs=args.epochs,
                                  criterion=nn.CrossEntropyLoss(label_smoothing=0.1))
        model = models.build_mobilenet_v2()
        opt = build_optimizer(model, head_names=("classifier.",))
        return run_experiment("E_mobilenet_v2", model, opt, train_loader, val_loader,
                              device, epochs=args.epochs)

    keys = ["baseline", "frozen", "resnet", "regularized", "mobilenet"] \
        if args.experiment == "all" else [args.experiment]
    for key in keys:
        results[key] = run(key)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / "experiments.json"
    json.dump(results, open(out, "w"), indent=2)
    print("results saved:", out)


if __name__ == "__main__":
    main()
