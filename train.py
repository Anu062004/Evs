"""Train and evaluate the pretrained ResNet-50 baseline.

Usage:
    python train.py --manifest data/manifest/manifest.csv --data-dir data/raw

Smoke test on a small sample (run this first):
    python train.py --manifest data/manifest/manifest.csv --data-dir data/raw \
        --epochs 1 --limit-per-class 10 --out-dir runs/smoke

Writes to --out-dir:
    train.log             full run log
    history.csv           per-epoch loss, Top-1 and macro-F1 for train/val
    curves.png            loss and macro-F1 curves
    best.pt               weights with the best validation macro-F1
    test_metrics.json     Top-1, Top-5, macro-F1 on the held-out test split
    confusion_matrix.png  / confusion_matrix.csv
    test_predictions.csv  top-3 species and confidence for every test image
"""

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from common import (ManifestDataset, build_model, build_transforms, confusion_matrix,
                    get_device, macro_f1, read_manifest)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """One pass over the loader. Trains when an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_logits, all_targets = [], []
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * len(targets)
            all_logits.append(logits.detach().float().cpu())
            all_targets.append(targets.cpu())
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    num_classes = logits.shape[1]
    top5 = logits.topk(min(5, num_classes), dim=1).indices
    cm = confusion_matrix(targets.numpy(), logits.argmax(1).numpy(), num_classes)
    return {
        "loss": total_loss / len(targets),
        "top1": float(np.diag(cm).sum() / cm.sum()),
        "top5": float((top5 == targets[:, None]).any(1).float().mean()),
        "macro_f1": macro_f1(cm),
        "cm": cm,
        "probs": logits.softmax(1),
    }


def set_backbone_trainable(model, trainable):
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = trainable


def plot_curves(history, path):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in zip(axes, ("loss", "macro_f1"), ("Loss", "Macro-F1")):
        ax.plot(epochs, [h[f"train_{key}"] for h in history], marker="o", label="train")
        ax.plot(epochs, [h[f"val_{key}"] for h in history], marker="o", label="val")
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, classes, path):
    size = max(6, 0.35 * len(classes))
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="runs/resnet50")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--freeze-epochs", type=int, default=2, help="epochs training only the new head")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="head learning rate; backbone uses lr/10")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit-per-class", type=int, help="cap images per class per split (smoke test)")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "train.log", "w")

    def log(message):
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    rows = read_manifest(args.manifest)
    classes = sorted({r["label"] for r in rows})
    by_split = defaultdict(list)
    per_class = defaultdict(int)
    for row in rows:
        key = (row["split"], row["label"])
        if args.limit_per_class and per_class[key] >= args.limit_per_class:
            continue
        per_class[key] += 1
        by_split[row["split"]].append(row)

    device = get_device()
    log(f"args: {vars(args)}")
    log(f"device: {device} | classes: {len(classes)} | "
        + " | ".join(f"{s}: {len(by_split[s])}" for s in ("train", "val", "test")))
    for split in ("train", "val", "test"):
        if not by_split[split]:
            raise SystemExit(f"The '{split}' split is empty; check the manifest")

    def loader(split, train):
        dataset = ManifestDataset(by_split[split], args.data_dir, classes, build_transforms(train))
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=train, num_workers=args.workers,
                          pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)

    train_loader = loader("train", True)
    val_loader = loader("val", False)

    model = build_model(len(classes), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": args.lr / 10},
        {"params": model.fc.parameters(), "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history = []
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        set_backbone_trainable(model, epoch > args.freeze_epochs)
        train = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch,
                        **{f"train_{k}": train[k] for k in ("loss", "top1", "macro_f1")},
                        **{f"val_{k}": val[k] for k in ("loss", "top1", "macro_f1")}})
        log(f"epoch {epoch:02d}/{args.epochs} | train loss {train['loss']:.4f} f1 {train['macro_f1']:.4f} | "
            f"val loss {val['loss']:.4f} top1 {val['top1']:.4f} f1 {val['macro_f1']:.4f} | "
            f"{time.time() - start:.0f}s")
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            torch.save({"model_state": model.state_dict(), "classes": classes, "epoch": epoch,
                        "val_macro_f1": best_f1}, out_dir / "best.pt")

    with open(out_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plot_curves(history, out_dir / "curves.png")

    # Final evaluation on the held-out test split, using the best validation weights.
    best = torch.load(out_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model_state"])
    test = run_epoch(model, loader("test", False), criterion, device)
    metrics = {"best_epoch": best["epoch"], "val_macro_f1": best["val_macro_f1"],
               "test_loss": test["loss"], "test_top1": test["top1"], "test_top5": test["top5"],
               "test_macro_f1": test["macro_f1"], "test_images": len(by_split["test"]),
               "classes": len(classes)}
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"test: {json.dumps(metrics)}")

    np.savetxt(out_dir / "confusion_matrix.csv", test["cm"], fmt="%d", delimiter=",",
               header=",".join(classes), comments="")
    plot_confusion_matrix(test["cm"], classes, out_dir / "confusion_matrix.png")

    k = min(3, len(classes))
    confidences, indices = test["probs"].topk(k, dim=1)
    with open(out_dir / "test_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true"] + [x for i in range(1, k + 1) for x in (f"pred{i}", f"conf{i}")])
        for row, conf, idx in zip(by_split["test"], confidences.tolist(), indices.tolist()):
            writer.writerow([row["path"], row["label"]]
                            + [x for c, i in zip(conf, idx) for x in (classes[i], f"{c:.4f}")])
    log(f"Saved weights, curves, metrics and predictions to {out_dir}")
    log_file.close()


if __name__ == "__main__":
    main()
