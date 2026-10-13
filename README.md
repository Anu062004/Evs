# Deep Learning-Based Bird Species Recognition

ResNet-50 baseline that identifies a bird species from a photo, with a small Flask website.
Environmental Science Project (BAIP508), Department of AI&ML.

## Results

Trained on 25 species from CUB-200-2011 (1,050 train / 225 validation / 225 test images).
On the held-out test split: Top-1 87.6%, Top-5 98.7%, macro-F1 0.878.
The full evidence is in `runs/resnet50/` (log, curves, confusion matrix, per-image predictions)
and `data/manifest/` (split manifest, class counts, rejected files).

## Run the website

    pip install -r requirements.txt
    python app.py

Then open http://127.0.0.1:5000.

## Retrain the model

    pip install -r requirements-train.txt
    python prepare_data.py --data-dir data/raw --max-classes 25
    python train.py --manifest data/manifest/manifest.csv --data-dir data/raw
    python export_onnx.py --weights runs/resnet50/best.pt

`data/raw` holds one folder per species. The dataset itself is not in this repository.
