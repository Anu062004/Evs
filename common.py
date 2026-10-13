"""Shared pieces for the bird species ResNet-50 baseline."""

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms(train):
    normalise = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        # Augmentation is applied to the training split only.
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalise,
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        normalise,
    ])


def build_model(num_classes, pretrained=True):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def read_manifest(manifest_path):
    with open(manifest_path, newline="") as f:
        return list(csv.DictReader(f))


class ManifestDataset(Dataset):
    """Images listed in the manifest for one split."""

    def __init__(self, rows, data_dir, classes, transform):
        self.rows = rows
        self.data_dir = Path(data_dir)
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image = Image.open(self.data_dir / row["path"]).convert("RGB")
        return self.transform(image), self.class_to_idx[row["label"]]


def load_checkpoint(weights_path, device):
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    model = build_model(len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt["classes"]


def confusion_matrix(targets, preds, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (targets, preds), 1)
    return cm


def macro_f1(cm):
    tp = np.diag(cm).astype(float)
    denom = cm.sum(axis=0) + cm.sum(axis=1)  # 2*tp + fp + fn
    seen = denom > 0  # ignore classes absent from both targets and predictions
    return float((2 * tp[seen] / denom[seen]).mean()) if seen.any() else 0.0
