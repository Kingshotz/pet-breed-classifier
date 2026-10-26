"""Grad-CAM: სურათის რომელ ნაწილს ეყრდნობა მოდელი პროგნოზისას.

გაშვება:
    python -m src.gradcam --checkpoint models/E_mobilenet_v2_best.pth
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src import config, models
from src.data import build_dataloaders
from src.train import get_device


class GradCAM:
    """ბოლო კონვოლუციური ფენის აქტივაციებს ვწონით კლასის გრადიენტებით."""

    def __init__(self, model, target_layer, device):
        self.model = model.eval()
        self.device = device
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor, class_idx=None):
        x = image_tensor.unsqueeze(0).to(self.device).requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
                            mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        confidence = float(F.softmax(logits, 1)[0, class_idx])
        return cam, class_idx, confidence


def denormalize(tensor):
    mean = torch.tensor(config.MEAN).view(3, 1, 1)
    std = torch.tensor(config.STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(config.MODELS_DIR / "E_mobilenet_v2_best.pth"))
    parser.add_argument("--num-samples", type=int, default=6)
    args = parser.parse_args()

    device = get_device()
    _, _, test_loader, class_names = build_dataloaders()
    test_set = test_loader.dataset

    model = models.build_mobilenet_v2(pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    cam_builder = GradCAM(model, model.features[-1], device)

    indices = np.linspace(0, len(test_set) - 1, args.num_samples).astype(int)
    fig, axes = plt.subplots(2, len(indices), figsize=(3 * len(indices), 6))
    for col, idx in enumerate(indices):
        image, label = test_set[int(idx)]
        cam, pred, confidence = cam_builder(image)
        axes[0, col].imshow(denormalize(image)); axes[0, col].axis("off")
        axes[0, col].set_title(f"true: {class_names[label]}", fontsize=8)
        axes[1, col].imshow(denormalize(image))
        axes[1, col].imshow(cam, cmap="jet", alpha=0.45); axes[1, col].axis("off")
        axes[1, col].set_title(f"pred: {class_names[pred]} ({confidence:.0%})", fontsize=8,
                               color="green" if pred == label else "red")
    plt.tight_layout()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / "gradcam.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
