"""Data QA and leakage-controlled 70:15:15 split.

Expects one sub-folder per species:

    data/raw/<species name>/<image files>

Usage:
    python prepare_data.py --data-dir data/raw --out-dir data/manifest \
        --source "Indian-Birds (Kaggle)" --licence "CC0"

Writes to --out-dir:
    manifest.csv        path, label, group, split  (input to train.py)
    class_counts.csv    per-class raw / kept / train / val / test counts
    rejected_files.csv  corrupt files and exact duplicates, with the reason
    dataset_report.json source, licence, seed and totals
"""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


def dhash(image, size=8):
    """Perceptual hash: survives resizing and re-compression of the same photo."""
    grey = image.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = grey.tobytes()
    bits = 0
    for row in range(size):
        for col in range(size):
            i = row * (size + 1) + col
            bits = (bits << 1) | (px[i] > px[i + 1])
    return bits


def near_duplicate_groups(files, label, known_groups, max_distance=4):
    """Group a class's images so near-duplicates (hashes within max_distance bits) stay together."""
    parent = list(range(len(files)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            if (files[i][1] ^ files[j][1]).bit_count() <= max_distance:
                parent[find(j)] = find(i)

    groups = defaultdict(list)
    for i, (rel, _) in enumerate(files):
        groups[known_groups.get(rel, f"{label}/{files[find(i)][1]:016x}")].append(rel)
    return groups


def load_groups(groups_csv):
    """Optional metadata CSV with columns path,group (photographer / session id)."""
    if not groups_csv:
        return {}
    with open(groups_csv, newline="") as f:
        return {row["path"]: row["group"] for row in csv.DictReader(f)}


def split_groups(groups, rng):
    """Assign whole groups to splits, filling whichever split is furthest below target."""
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: len(kv[1]), reverse=True)
    total = sum(len(paths) for _, paths in items)
    filled = Counter()
    assignment = {}
    for group, paths in items:
        split = max(SPLITS, key=lambda s: SPLITS[s] * total - filled[s])
        assignment[group] = split
        filled[split] += len(paths)
    return assignment


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="folder with one sub-folder per species")
    parser.add_argument("--out-dir", default="data/manifest")
    parser.add_argument("--groups-csv", help="optional CSV (path,group) with photographer/session ids")
    parser.add_argument("--max-classes", type=int, help="keep only the N largest classes (e.g. 25)")
    parser.add_argument("--source", default="unspecified", help="dataset source, recorded in the report")
    parser.add_argument("--licence", default="unspecified", help="dataset licence, recorded in the report")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    known_groups = load_groups(args.groups_csv)
    rng = random.Random(args.seed)

    class_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class folders found in {data_dir}")

    raw_counts = Counter()
    rejected = []
    seen_md5 = {}
    hashed = defaultdict(list)  # label -> [(path, perceptual hash)]

    for class_dir in class_dirs:
        label = class_dir.name
        files = sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        raw_counts[label] = len(files)
        for path in files:
            rel = path.relative_to(data_dir).as_posix()
            try:
                with Image.open(path) as image:
                    image.load()
                    phash = dhash(image)
            except Exception as error:
                rejected.append((rel, label, "corrupt", str(error)))
                continue
            md5 = hashlib.md5(path.read_bytes()).hexdigest()
            if md5 in seen_md5:
                rejected.append((rel, label, "exact_duplicate", seen_md5[md5]))
                continue
            seen_md5[md5] = rel
            hashed[label].append((rel, phash))

    kept = {label: near_duplicate_groups(files, label, known_groups) for label, files in hashed.items()}
    labels = sorted(kept)
    if args.max_classes:
        size = lambda l: sum(len(p) for p in kept[l].values())
        labels = sorted(sorted(labels, key=size, reverse=True)[: args.max_classes])

    manifest = []
    count_rows = []
    for label in labels:
        assignment = split_groups(kept[label], rng)
        per_split = Counter()
        for group, paths in kept[label].items():
            for rel in paths:
                manifest.append((rel, label, group, assignment[group]))
                per_split[assignment[group]] += 1
        n_kept = sum(per_split.values())
        count_rows.append((label, raw_counts[label], n_kept, len(kept[label]),
                           per_split["train"], per_split["val"], per_split["test"]))
        if not (per_split["train"] and per_split["val"] and per_split["test"]):
            print(f"WARNING: class '{label}' has an empty split ({dict(per_split)}); it needs more images")

    with open(out_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "group", "split"])
        writer.writerows(sorted(manifest))
    with open(out_dir / "class_counts.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "raw", "kept", "groups", "train", "val", "test"])
        writer.writerows(count_rows)
    with open(out_dir / "rejected_files.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "reason", "detail"])
        writer.writerows(rejected)

    split_totals = Counter(row[3] for row in manifest)
    report = {
        "source": args.source,
        "licence": args.licence,
        "seed": args.seed,
        "classes": len(labels),
        "raw_images": sum(raw_counts.values()),
        "kept_images": len(manifest),
        "rejected": dict(Counter(r[2] for r in rejected)),
        "split_counts": dict(split_totals),
        "leakage_control": "photographer/session ids from --groups-csv" if known_groups
                           else "near-duplicate (perceptual hash) groups only; no photographer/session metadata supplied",
    }
    (out_dir / "dataset_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote manifest and QA reports to {out_dir}")


if __name__ == "__main__":
    main()
