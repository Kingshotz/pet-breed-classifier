"""მონაცემების ჩატვირთვა, augmentation და train/val/test გაყოფა."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from src import config


def build_transforms():
    """Train-ზე augmentation, val/test-ზე დეტერმინისტული დამუშავება.

    val/test-ზე შემთხვევითობა არ გვინდა, თორემ ერთი და იგივე მოდელი
    ორჯერ გაზომვისას სხვადასხვა შედეგს მოგვცემს.
    """
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])
    return train_tf, eval_tf


def build_dataloaders(root=None, batch_size=None, download=True):
    """აბრუნებს (train_loader, val_loader, test_loader, class_names).

    trainval ორჯერ იტვირთება, რომ train-ს augmentation ჰქონდეს,
    val-ს კი — არა, მიუხედავად იმისა, რომ ერთი და იმავე ფაილებიდან მოდის.
    """
    root = str(root or config.DATA_DIR)
    batch_size = batch_size or config.BATCH_SIZE
    train_tf, eval_tf = build_transforms()

    full_train = OxfordIIITPet(root=root, split="trainval", transform=train_tf, download=download)
    full_val = OxfordIIITPet(root=root, split="trainval", transform=eval_tf, download=download)
    test_set = OxfordIIITPet(root=root, split="test", transform=eval_tf, download=download)

    n = len(full_train)
    n_train = int((1 - config.VAL_SPLIT) * n)
    generator = torch.Generator().manual_seed(config.SEED)
    indices = torch.randperm(n, generator=generator).tolist()

    train_set = Subset(full_train, indices[:n_train])
    val_set = Subset(full_val, indices[n_train:])

    loaders = (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS),
    )
    return (*loaders, test_set.classes)
