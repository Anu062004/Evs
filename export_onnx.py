"""Export the trained weights to ONNX for the web app.

The web app runs on onnxruntime so it can be hosted without PyTorch.

Usage:
    python export_onnx.py --weights runs/resnet50/best.pt

Writes model/bird_resnet50.onnx and model/classes.json.
"""

import argparse
import json
from pathlib import Path

import torch

from common import IMAGE_SIZE, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="runs/resnet50/best.pt")
    parser.add_argument("--out-dir", default="model")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, classes = load_checkpoint(args.weights, torch.device("cpu"))
    example = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(model, example, out_dir / "bird_resnet50.onnx", input_names=["image"],
                      output_names=["logits"], dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      dynamo=False)
    (out_dir / "classes.json").write_text(json.dumps(classes, indent=2))
    print(f"Wrote {out_dir / 'bird_resnet50.onnx'} and classes.json ({len(classes)} classes)")


if __name__ == "__main__":
    main()
