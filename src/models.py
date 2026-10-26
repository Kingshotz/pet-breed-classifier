"""მოდელის არქიტექტურები: baseline CNN და pretrained backbone-ები."""
import torch.nn as nn
from torchvision import models

from src import config


def conv_block(in_ch, out_ch):
    """Conv -> BatchNorm -> ReLU -> MaxPool.

    BatchNorm ვარჯიშს ასტაბილურებს, ReLU არაწრფივობას ამატებს
    (მის გარეშე ფენების ჯაჭვი ერთ წრფივ გარდაქმნად იშლებოდა),
    MaxPool კი სივრცით ზომას ანახევრებს.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """Baseline: ნულიდან გაწვრთნილი ~399K პარამეტრიანი ქსელი."""

    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32), conv_block(32, 64),
            conv_block(64, 128), conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def build_resnet18(num_classes=config.NUM_CLASSES, pretrained=True, freeze_backbone=False):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
    return model


def build_mobilenet_v2(num_classes=config.NUM_CLASSES, pretrained=True):
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
