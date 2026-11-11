"""
Vision Core IA — Application Web d'Inspection Visuelle et Détection d'Anomalies
Déployable sur Streamlit Community Cloud.
"""

import os

# 1. Variables d'environnement bas niveau (Anti-OOM / Restriction CPU)
os.environ["MALLOC_TRIM_THRESHOLD_"] = "100000"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ.setdefault("TRUST_REMOTE_CODE", "1")

import io
import re
import gc
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import gdown
from PIL import Image
import streamlit as st

# Limitation PyTorch CPU
torch.set_num_threads(1)

from anomalib.deploy import TorchInferencer

# -----------------------------------------------------------------------------
# 1. CONFIGURATION DES CATÉGORIES ET IDENTIFIANTS GOOGLE DRIVE
# -----------------------------------------------------------------------------
DRIVE_MODEL_IDS = {
    "metal_nut": "1I9lZY-d0ksDkVsk6EPXh0YS2A_SZwueL",
    "transistor": "1L1I3KDNQS_lIj_Xuf--u0x4SudEQRO2J",
    "carpet": "13bwkq2S8ZWhTuyiKVTlyOAlTbFB2-M7E",
}

CATEGORIES = {
    "metal_nut": {"label": "Écrou en métal (metal_nut)", "threshold": 0.62},
    "transistor": {"label": "Transistor (transistor)", "threshold": 0.60},
    "carpet": {"label": "Textile (carpet)", "threshold": 0.56},
}

MODELS_DIR = Path(__file__).resolve().parent / "models"


# -----------------------------------------------------------------------------
# 2. TÉLÉCHARGEMENT ET INFÉRENCE CPU
# -----------------------------------------------------------------------------
def extract_gdrive_id(url_or_id: str) -> str:
    match = re.search(r'[-\w]{25,}', url_or_id)
    return match.group(0) if match else url_or_id


def ensure_model_downloaded(category: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_file = MODELS_DIR / f"{category}.pt"

    if not model_file.exists():
        raw_val = DRIVE_MODEL_IDS.get(category, "")
        file_id = extract_gdrive_id(raw_val)
        
        if file_id:
            print(f"Téléchargement du modèle {category} (ID: {file_id})...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(model_file), quiet=False)
        else:
            raise FileNotFoundError(
                f"Modèle introuvable localement pour '{category}'."
            )

    return model_file


@dataclass
class Result:
    score: float
    is_defective: bool
    anomaly_map: np.ndarray
    mask: np.ndarray
    latency_ms: float


class Inspector:
    def __init__(self, category: str) -> None:
        path = ensure_model_downloaded(category)
        self.inferencer = TorchInferencer(path=path, device="cpu")
        self.device = "cpu"

    def predict(self, image: Image.Image, sensitivity: float = 0.55) -> Result:
        image = image.convert("RGB")
        w, h = image.size

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.inferencer.predict(image)
        latency = (time.perf_counter() - t0) * 1000

        score = float(out.pred_score.flatten()[0])
        amap = out.anomaly_map.detach().cpu().numpy().squeeze().astype(np.float32)
        amap = cv2.resize(np.clip(amap, 0, 1), (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (amap >= sensitivity).astype(np.uint8)
        return Result(score, score >= sensitivity, amap, mask, latency)


def overlay(image: Image.Image, amap: np.ndarray, alpha: float = 0.50) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    heat = cv2.applyColorMap((np.clip(amap, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2.addWeighted(rgb, 1 - alpha, heat, alpha, 0))


def draw_contours(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, cnts, -1, (0, 229, 255), max(2, rgb.shape[0] // 180))
    return Image.fromarray(rgb)


# -----------------------------------------------------------------------------
# 3. INTERFACE STREAMLIT (DESIGN NOIR & BLEU TURQUOISE)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Vision Core IA | Inspection Industrielle",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {
        background-color: #0A0E17;
        color: #E2E8F0;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1E293B;
    }
    h1, h2, h3, h4 {
        color: #00E5FF !important;
        font-weight: 700 !important;
    }
    .metric-card {
        background: #151D2A;
        border: 1px solid #1E293B;
        border-left: 4px solid #00E5FF;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.85rem;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-value {
        color: #00E5FF;
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 4px;
    }
    .badge-pass {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid #10B981;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-fail {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid #EF4444;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(135deg, #00E5FF 0%, #0088FF 100%) !important;
        color: #050B14 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius:
            
