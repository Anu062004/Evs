"""Predict the top-3 species for one or more images.

Usage:
    python predict.py --weights runs/resnet50/best.pt photo1.jpg photo2.jpg
"""

import argparse

import torch
from PIL import Image

from common import build_transforms, get_device, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    device = get_device()
    model, classes = load_checkpoint(args.weights, device)
    transform = build_transforms(train=False)
    k = min(args.top_k, len(classes))

    for path in args.images:
        image = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model(image).softmax(1)[0]
        confidences, indices = probs.topk(k)
        print(path)
        for rank, (conf, idx) in enumerate(zip(confidences.tolist(), indices.tolist()), 1):
            print(f"  {rank}. {classes[idx]:<30} {conf:.1%}")


if __name__ == "__main__":
    main()
