"""Web app for the bird species recogniser.

Usage:
    python app.py            # then open http://127.0.0.1:5000

Runs the ONNX model written by export_onnx.py, so the site needs no PyTorch.
"""

import base64
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from flask import Flask, abort, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).parent
WEIGHTS = ROOT / "model/bird_resnet50.onnx"
CLASSES = ROOT / "model/classes.json"
RUN_DIR = ROOT / "runs/resnet50"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RUN_FILES = {"curves.png", "confusion_matrix.png"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

# The pages of the site, in navigation order: (endpoint, label, status).
SITEMAP = [
    ("home", "Home", "live"),
    ("identify", "Identify", "live"),
    ("species", "Species", "live"),
    ("performance", "Model performance", "live"),
    ("biodiversity", "Biodiversity", "planned"),
    ("about", "About", "live"),
]


@app.context_processor
def inject_sitemap():
    return {"sitemap": SITEMAP}


@lru_cache(maxsize=1)
def get_model():
    session = ort.InferenceSession(str(WEIGHTS), providers=["CPUExecutionProvider"])
    return session, json.loads(CLASSES.read_text())


def preprocess(image):
    """Same steps as the validation transform in common.py: resize to 256, centre-crop 224, normalise."""
    width, height = image.size
    scale = 256 / min(width, height)
    image = image.resize((max(256, int(width * scale)), max(256, int(height * scale))), Image.BILINEAR)
    left = round((image.width - 224) / 2)
    top = round((image.height - 224) / 2)
    pixels = np.asarray(image.crop((left, top, left + 224, top + 224)), dtype=np.float32) / 255
    return ((pixels - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]


def display_name(label):
    """'001.Black_footed_Albatross' -> 'Black footed Albatross'"""
    return label.split(".", 1)[-1].replace("_", " ")


def predict(image, top_k=3):
    session, classes = get_model()
    logits = session.run(None, {"image": preprocess(image)})[0][0]
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    best = probs.argsort()[::-1][:top_k]
    return [{"species": display_name(classes[i]), "confidence": float(probs[i])} for i in best]


@app.route("/")
def home():
    return render_template("home.html", model_ready=WEIGHTS.exists())


@app.route("/identify", methods=["GET", "POST"])
def identify():
    context = {"model_ready": WEIGHTS.exists()}
    if request.method == "POST" and context["model_ready"]:
        upload = request.files.get("photo")
        if not upload or not upload.filename:
            context["error"] = "Choose a photo first."
        else:
            data = upload.read()
            try:
                image = Image.open(io.BytesIO(data)).convert("RGB")
            except (UnidentifiedImageError, OSError):
                context["error"] = "That file could not be read as an image."
            else:
                context["predictions"] = predict(image)
                preview = image.copy()
                preview.thumbnail((640, 640))
                buffer = io.BytesIO()
                preview.save(buffer, format="JPEG", quality=85)
                context["preview"] = base64.b64encode(buffer.getvalue()).decode()
                context["filename"] = upload.filename
    return render_template("identify.html", **context)


@app.route("/species")
def species():
    names = [display_name(c) for c in get_model()[1]] if WEIGHTS.exists() else []
    return render_template("species.html", species=names)


@app.route("/performance")
def performance():
    metrics_file = RUN_DIR / "test_metrics.json"
    metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else None
    plots = sorted(f for f in RUN_FILES if (RUN_DIR / f).exists())
    return render_template("performance.html", metrics=metrics, plots=plots)


@app.route("/run-files/<name>")
def run_file(name):
    if name not in RUN_FILES:
        abort(404)
    return send_from_directory(RUN_DIR, name)


@app.route("/biodiversity")
def biodiversity():
    return render_template("biodiversity.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.errorhandler(413)
def too_large(_error):
    return render_template("identify.html", model_ready=WEIGHTS.exists(),
                           error="That photo is larger than 10 MB."), 413


if __name__ == "__main__":
    app.run(debug=True)
