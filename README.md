# Deep Learning-Based Bird Species Recognition

Upload a photo of a bird and get the three most likely species, each with a confidence score.
The project has two parts: a training pipeline that fine-tunes a ResNet-50 network on bird
photographs, and a small website that runs the trained model.

**Live site:** https://bird-species-recognition.vercel.app

Environmental Science Project (BAIP508), Department of Artificial Intelligence and Machine Learning.
Team: Sonali N Gowda, Anubhav, Reyyi Shreyas, Mayank Prakash. Guide: Prof. Shobith Tembhre.

## What the site does

| Page | What it does |
|---|---|
| Home | Introduces the project and links to every page |
| Identify | Upload a photo, get the top-3 species with confidence |
| Species | Photos of the 25 species the model knows |
| Performance | Test accuracy, training curves and confusion matrix |
| Biodiversity | Planned for Phase 2, not built yet |
| About | Team and photo credits |

## Results

Trained on 25 species from CUB-200-2011, split 1,050 train / 225 validation / 225 test images.
On the held-out test split:

| Metric | Result |
|---|---|
| Top-1 accuracy | 87.6% |
| Top-5 accuracy | 98.7% |
| Macro-F1 | 0.878 |

Most errors are between the all-black species (Fish Crow, American Crow, Shiny Cowbird).
The model is a closed classifier: a bird outside the 25 species is still given one of those names.

The evidence behind these numbers is in `runs/resnet50/` (run log, curves, confusion matrix,
per-image predictions) and `data/manifest/` (split manifest, class counts, rejected files).

## How it works

**Training (offline, PyTorch)**

1. `prepare_data.py` scans a folder with one sub-folder per species. It rejects corrupt files and
   exact duplicates, groups near-duplicate images so they stay in one split, and writes a
   70:15:15 train/validation/test manifest.
2. `train.py` fine-tunes an ImageNet-pretrained ResNet-50 on 224 × 224 images, with augmentation on
   the training split only. It keeps the weights with the best validation macro-F1, then scores
   them once on the test split.
3. `export_onnx.py` converts the best weights to ONNX so the website can run without PyTorch.

**Website (Flask + ONNX Runtime)**

1. The browser posts the photo to `/identify`.
2. `app.py` resizes it to 256 px on the short side, centre-crops 224 × 224 and normalises it,
   the same steps used for validation during training.
3. The ONNX model returns one score per species; softmax turns the scores into confidences.
4. The page shows the three highest, each with a reference photo of that species.

Uploaded photos are processed in memory and are not saved.

## Project layout

    app.py               Flask website and inference
    templates/, static/  Pages, styles, script and photos
    model/               ONNX model and class list used by the site
    prepare_data.py      Data QA and split
    train.py             Training and evaluation
    export_onnx.py       PyTorch weights -> ONNX
    predict.py           Command-line prediction with the PyTorch weights
    common.py            Model and transforms shared by the training scripts
    runs/resnet50/       Results of the training run
    data/manifest/       Split manifest and data QA reports

## Run the website locally

    pip install -r requirements.txt
    python app.py

Then open http://127.0.0.1:5000.

## Retrain the model

    pip install -r requirements-train.txt
    python prepare_data.py --data-dir data/raw --max-classes 25
    python train.py --manifest data/manifest/manifest.csv --data-dir data/raw
    python export_onnx.py --weights runs/resnet50/best.pt

`data/raw` holds one folder per species. The dataset itself is not in this repository.

## Credits

- Training data: CUB-200-2011 (Caltech-UCSD Birds), used for non-commercial research and education.
- Photographs on the site: Wikimedia Commons, under the Creative Commons licences listed on the
  site's About page and in `static/img/credits.json`.
