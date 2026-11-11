# Vision Core IA — Application Web d'Inspection Visuelle et Détection d'Anomalies
# Déployable sur Streamlit Community Cloud.

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
# 1. CONFIGURATION DES CATÉGORIES ET SEUILS OPTIMAUX
# -----------------------------------------------------------------------------
DRIVE_MODEL_IDS = {
    "metal_nut": "1I9lZY-d0ksDkVsk6EPXh0YS2A_SZwueL",
    "transistor": "1L1I3KDNQS_lIj_Xuf--u0x4SudEQRO2J",
    "carpet": "13bwkq2S8ZWhTuyiKVTlyOAlTbFB2-M7E",
}

# Seuils de décision calibrés par catégorie
CATEGORIES = {
    "metal_nut": {"label": "Écrou en métal (metal_nut)", "threshold": 0.62},
    "transistor": {"label": "Transistor (transistor)", "threshold": 0.60},
    "carpet": {"label": "Textile (carpet)", "threshold": 0.56},
}

MODELS_DIR = Path(__file__).resolve().parent / "models"


# -----------------------------------------------------------------------------
# 2. TÉLÉCHARGEMENT ET LOGIQUE D'INFÉRENCE CPU
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

        # Score global d'anomalie de l'image
        score = float(out.pred_score.flatten()[0])
        
        # Carte d'anomalie au niveau pixel
        amap_raw = out.anomaly_map.detach().cpu().numpy().squeeze().astype(np.float32)
        amap_resized = cv2.resize(amap_raw, (w, h), interpolation=cv2.INTER_LINEAR)

        # Découpage sur l'échelle absolue (0.0 à 1.0) sans déformation Min-Max
        amap_vis = np.clip(amap_resized, 0.0, 1.0)

        # Décision de conformité globale basée sur le score de l'image
        is_defective = score >= sensitivity

        # Masque binaire localisé basé uniquement sur les pixels dépassant le seuil
        mask = (amap_vis >= sensitivity).astype(np.uint8)

        return Result(score, is_defective, amap_vis, mask, latency)


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

css_lines = [
    "<style>",
    "    .stApp { background-color: #0A0E17; color: #E2E8F0; font-family: 'Inter', system-ui, -apple-system, sans-serif; }",
    "    [data-testid='stSidebar'] { background-color: #111827; border-right: 1px solid #1E293B; }",
    "    h1, h2, h3, h4 { color: #00E5FF !important; font-weight: 700 !important; }",
    "    .metric-card { background: #151D2A; border: 1px solid #1E293B; border-left: 4px solid #00E5FF; border-radius: 10px; padding: 16px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3); }",
    "    .metric-title { color: #94A3B8; font-size: 0.85rem; text-transform: uppercase; font-weight: 600; }",
    "    .metric-value { color: #00E5FF; font-size: 1.8rem; font-weight: 800; margin-top: 4px; }",
    "    .badge-pass { background-color: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid #10B981; padding: 10px 20px; border-radius: 8px; font-weight: 700; font-size: 1.1rem; display: inline-block; }",
    "    .badge-fail { background-color: rgba(239, 68, 68, 0.15); color: #EF4444; border: 1px solid #EF4444; padding: 10px 20px; border-radius: 8px; font-weight: 700; font-size: 1.1rem; display: inline-block; }",
    "    .stButton>button, .stDownloadButton>button { background: linear-gradient(135deg, #00E5FF 0%, #0088FF 100%) !important; color: #050B14 !important; font-weight: 700 !important; border: none !important; border-radius: 8px !important; padding: 10px 24px !important; box-shadow: 0 0 15px rgba(0, 229, 255, 0.3) !important; }",
    "</style>",
]
st.markdown("\n".join(css_lines), unsafe_allow_html=True)

# Barre Latérale
with st.sidebar:
    st.markdown("## ⚙️ Configuration Système")
    st.markdown("---")
    category = st.selectbox(
        "1. Matériel à inspecter",
        list(CATEGORIES),
        format_func=lambda c: CATEGORIES[c]["label"],
    )
    
    default_thresh = CATEGORIES[category]["threshold"]
    
    sensitivity = st.slider(
        "2. Seuil de Décision (Sensibilité)",
        min_value=0.10,
        max_value=0.90,
        value=default_thresh,
        step=0.01,
        help="Seuil calibré dynamiquement selon la catégorie pour éviter les faux rejets.",
    )
    st.markdown("---")
    st.markdown("### ⚡ Spécifications Modèle")
    st.markdown("- **Algorithme** : PatchCore (WideResNet-50)")
    st.markdown("- **Mode** : Non supervisé")
    st.markdown("- **Coreset Ratio** : 20%")

# En-tête Principal
st.title("🔬 VISION CORE IA")
st.markdown("<h4 style='color: #94A3B8 !important; margin-top: -15px;'>Inspection Visuelle Automatisée & Explicable par Deep Learning</h4>", unsafe_allow_html=True)
st.divider()

# Gestion mémoire : 1 seul modèle à la fois en RAM
if "current_category" not in st.session_state:
    st.session_state["current_category"] = None
    st.session_state["inspector"] = None

if st.session_state["current_category"] != category:
    st.session_state["inspector"] = None
    gc.collect()
    
    try:
        with st.spinner(f"⚡ Chargement optimisé du modèle {category}..."):
            st.session_state["inspector"] = Inspector(category)
            st.session_state["current_category"] = category
    except Exception as e:
        st.error(f"❌ Erreur d'initialisation : {e}")
        st.stop()

inspector = st.session_state["inspector"]
st.sidebar.success(f"🧠 Modèle `{category}` actif (CPU)")

# Disposition principale
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📷 Acquisition de l'Image")
    uploaded_file = st.file_uploader(
        f"Chargez une image pour la catégorie '{category}'",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Image d'entrée reçue", use_container_width=True)
    else:
        st.info("💡 Veuillez sélectionner une image pour démarrer l'analyse en temps réel.")

with col_right:
    st.markdown("### 📊 Diagnostic & Décision IA")
    if not uploaded_file:
        st.warning("En attente de l'acquisition d'une image...")
    else:
        res = inspector.predict(image, sensitivity=sensitivity)
        
        if res.is_defective:
            html_fail = f'<div class="badge-fail">❌ PIÈCE DÉFECTUEUSE (Score: {res.score:.3f})</div>'
            st.markdown(html_fail, unsafe_allow_html=True)
        else:
            html_pass = f'<div class="badge-pass">✅ PIÈCE CONFORME (Score: {res.score:.3f})</div>'
            st.markdown(html_pass, unsafe_allow_html=True)
            
        st.write("")
        m1, m2, m3 = st.columns(3)
        with m1:
            card1 = f'<div class="metric-card"><div class="metric-title">Score Anomalie</div><div class="metric-value">{res.score:.2f}</div></div>'
            st.markdown(card1, unsafe_allow_html=True)
        with m2:
            card2 = f'<div class="metric-card"><div class="metric-title">Surface Impactée</div><div class="metric-value">{100 * res.mask.mean():.1f}%</div></div>'
            st.markdown(card2, unsafe_allow_html=True)
        with m3:
            card3 = f'<div class="metric-card"><div class="metric-title">Latence</div><div class="metric-value">{res.latency_ms:.0f} ms</div></div>'
            st.markdown(card3, unsafe_allow_html=True)
            
        st.write("")
        st.markdown("#### 🔍 Explicabilité Visuelle (XAI)")
        tab1, tab2 = st.tabs(["🔥 Carte de Chaleur (Heatmap)", "🔴 Contours Localisés"])
        with tab1:
            st.image(overlay(image, res.anomaly_map), use_container_width=True)
        with tab2:
            st.image(draw_contours(image, res.mask), use_container_width=True)
            
        buf = io.BytesIO()
        overlay(image, res.anomaly_map).save(buf, format="PNG")
        st.download_button(
            label="⬇️ Télécharger le Rapport Visuel",
            data=buf.getvalue(),
            file_name=f"rapport_{category}.png",
            mime="image/png",
        )
