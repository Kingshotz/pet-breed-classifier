"""ერთ ადგილას თავმოყრილი პარამეტრები — ექსპერიმენტების განმეორებადობისთვის."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

SEED = 42
NUM_CLASSES = 37
IMAGE_SIZE = 224
BATCH_SIZE = 32
VAL_SPLIT = 0.2
NUM_WORKERS = 2

# ImageNet-ის სტატისტიკა: pretrained მოდელები ამ ნორმალიზაციაზე გაიწვრთნენ
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
EPOCHS = 6
