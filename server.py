import io, os, json
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import torch
import torch.nn as nn
from torchvision import transforms, models
from sklearn.base import BaseEstimator
from typing import Dict, List

# ---------- Config ----------
PCA_PATH = os.getenv("PCA_PATH", "pca.pkl")
RF_PATH  = os.getenv("RF_PATH",  "model.pkl")       # RandomForest (optional)
ANN_PATH = os.getenv("ANN_PATH", "ann_model.pt")    # ANN (optional)
LABELS_PATH = os.getenv("LABELS_PATH", "labels.json")
IMAGE_SIZE = 224  # must match training
DEVICE = torch.device("cpu")  # keep CPU for portability

# ---------- App ----------
app = FastAPI(title="DR Classifier Server")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ---------- Load label map ----------
with open(LABELS_PATH, "r") as f:
    label_map: Dict[str,int] = json.load(f)
# index -> label list
inv_labels: List[str] = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

# ---------- Preprocessing (must match training) ----------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),                        # 0..1
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # -> [-1,1]
])

# ---------- Feature extractor (ResNet18, fc as Identity) ----------
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = nn.Identity()
resnet.eval().to(DEVICE)
torch.set_grad_enabled(False)

# ---------- Load PCA + classifiers ----------
pca = joblib.load(PCA_PATH)  # type: ignore

rf_model: BaseEstimator
if os.path.exists(RF_PATH):
    rf_model = joblib.load(RF_PATH)  # type: ignore

class SimpleANN(nn.Module):
    def __init__(self, input_dim=100, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    def forward(self, x): return self.net(x)

ann_model: SimpleANN
if os.path.exists(ANN_PATH):
    ann_model = SimpleANN(input_dim=pca.n_components_, num_classes=len(inv_labels)).to(DEVICE)
    ann_model.load_state_dict(torch.load(ANN_PATH, map_location=DEVICE))
    ann_model.eval()

def extract_features(pil: Image.Image) -> np.ndarray:
    """Image -> ResNet18 512-dim feature."""
    x = transform(pil).unsqueeze(0).to(DEVICE)        # [1,3,224,224]
    feats = resnet(x).cpu().numpy().squeeze()         # [512]
    return feats

def topk(probs: np.ndarray, k: int = 5):
    idx = np.argsort(-probs)[:k]
    return [{"index": int(i), "label": inv_labels[int(i)], "prob": float(probs[i])} for i in idx]

@app.get("/health")
def health():
    return {
        "ok": True,
        "has_rf": rf_model is not None,
        "has_ann": ann_model is not None,
        "pca_components": int(getattr(pca, "n_components_", 0)),
        "classes": inv_labels,
        "image_size": IMAGE_SIZE
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...), clf: str = Query("ann", enum=["ann","rf"])):
    try:
        raw = await file.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image")

    # 1) ResNet18 features
    feats = extract_features(pil)             # [512]

    # 2) PCA
    z = pca.transform(feats.reshape(1, -1))   # [1, 100]

    # 3) Classifier
    if clf == "rf":
        if rf_model is None: raise HTTPException(400, "RF model not available")
        probs = rf_model.predict_proba(z)[0]  # sklearn probs
    else:
        if ann_model is None: raise HTTPException(400, "ANN model not available")
        with torch.no_grad():
            logits = ann_model(torch.from_numpy(z.astype(np.float32)).to(DEVICE))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return {"top": topk(probs, k=min(5, len(inv_labels))), "raw": probs.tolist()}
