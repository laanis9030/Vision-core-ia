"""
Vision Core IA — Application Web d'Inspection Visuelle et Détection d'Anomalies
Déployable directement sur Streamlit Community Cloud.
"""

import os
import io
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import gdown
from PIL import Image
import streamlit as st

# Activation de la provenance de confiance pour Anomalib TorchInferencer
os.environ.setdefault("TRUST_REMOTE_CODE", "1")
from anomalib.deploy import TorchInferencer

# -----------------------------------------------------------------------------
# 1. CONFIGURATION DES CATÉGORIES ET IDENTIFIANTS GOOGLE DRIVE
# -----------------------------------------------------------------------------
# Remplacez les chaînes ci-dessous par les IDs réels de vos fichiers .pt sur Google Drive
# Exemple : pour le lien https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9/view -> ID = "1a2b3c4d5e6f7g8h9"
DRIVE_MODEL_IDS = {
    "metal_nut": "https://drive.google.com/file/d/13bwkq2S8ZWhTuyiKVTlyOAlTbFB2-M7E/view?usp=drive_link",
    "transistor": "https://drive.google.com/file/d/1I9lZY-d0ksDkVsk6EPXh0YS2A_SZwueL/view?usp=drive_link",
    "carpet": "https://drive.google.com/file/d/1L1I3KDNQS_lIj_Xuf--u0x4SudEQRO2J/view?usp=drive_link",
}

CATEGORIES = {
    "metal_nut": {"label": "Écrou en métal (metal_nut)"},
    "transistor": {"label": "Transistor (transistor)"},
    "carpet": {"label": "Textile (carpet)"},
}

MODELS_DIR = Path(__file__).resolve().parent / "models"


# -----------------------------------------------------------------------------
# 2. GESTION DES MODÈLES & INFÉRENCE
# -----------------------------------------------------------------------------
@dataclass
class Result:
    score: float
    is_defective: bool
    anomaly_map: np.ndarray
    mask: np.ndarray
    latency_ms: float


def ensure_model_downloaded(category: str) -> Path:
    """Télécharge le modèle .pt depuis Google Drive s'il n'est pas présent localement."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_file = MODELS_DIR / f"{category}.pt"

    if not model_file.exists():
        file_id = DRIVE_MODEL_IDS.get(category, "")
        if file_id and not file_id.startswith("VOTRE_FILE_ID"):
            url = f"https://drive.google.com/uc?id={file_id}"
            st.toast(f"📥 Téléchargement du modèle {category} depuis Google Drive...", icon="📦")
            gdown.download(url, str(model_file), quiet=False)
        else:
            raise FileNotFoundError(
                f"Modèle introuvable localement pour '{category}' et aucun ID Google Drive valide renseigné dans app.py."
            )

    return model_file


class Inspector:
    def __init__(self, category: str, device: str = "auto") -> None:
        path = ensure_model_downloaded(category)
        self.inferencer = TorchInferencer(path=path, device=device)
        self.device = str(self.inferencer.device)

    def predict(self, image: Image.Image, sensitivity: float = 0.55) -> Result:
        image = image.convert("RGB")
        w, h = image.size

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.inferencer.predict(image)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = (time.perf_counter() - t0) * 1000

        score = float(out.pred_score.flatten()[0])
        amap = out.anomaly_map.detach().cpu().numpy().squeeze().astype(np.float32)
        amap = cv2.resize(np.clip(amap, 0, 1), (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (amap >= sensitivity).astype(np.uint8)
        return Result(score, score >= sensitivity, amap, mask, latency)


def overlay(image: Image.Image, amap: np.ndarray, alpha: float = 0.50) -> Image.Image:
    """Superpose la carte de chaleur thermique sur l'image d'origine."""
    rgb = np.array(image.convert("RGB"))
    heat = cv2.applyColorMap((np.clip(amap, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2.addWeighted(rgb, 1 - alpha, heat, alpha, 0))


def draw_contours(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Entoure les zones défectueuses d'un contour bleu turquoise néon."""
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

# Injection CSS Personnalisée (Thème Noir Deep & Bleu Turquoise #00E5FF)
st.markdown(
    """
<style>
    /* Fond principal et typographie */
    .stApp {
        background-color: #0A0E17;
        color: #E2E8F0;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Barre latérale */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1E293B;
    }
    
    /* Titres principal et secondaires */
    h1, h2, h3, h4 {
        color: #00E5FF !important;
        font-weight: 700 !important;
    }
    
    /* Cartes de métriques */
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
    
    /* Badges de statut */
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
    
    /* Boutons personnalisés */
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(135deg, #00E5FF 0%, #0088FF 100%) !important;
        color: #050B14 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.3) !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_inspector_cached(cat: str) -> Inspector:
    return Inspector(cat)


# Barre Latérale
with st.sidebar:
    st.markdown("## ⚙️ Configuration Système")
    st.markdown("---")
    category = st.selectbox(
        "1. Matériel à inspecter",
        list(CATEGORIES),
        format_func=lambda c: CATEGORIES[c]["label"],
    )
    sensitivity = st.slider(
        "2. Seuil de Décision (Sensibilité)",
        min_value=0.10,
        max_value=0.90,
        value=0.55,
        step=0.05,
        help="0,55 = Seuil optimal calibré pour supprimer les faux positifs sur le métal.",
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

# Chargement du modèle avec cache
try:
    with st.spinner("⚡ Initialisation du réseau d'inférence..."):
        inspector = load_inspector_cached(category)
    st.sidebar.success(f"🧠 Modèle `{category}` actif ({inspector.device})")
except Exception as e:
    st.error(f"❌ Erreur de chargement du modèle : {e}")
    st.info("Assurez-vous d'avoir renseigné les bons IDs Google Drive dans `DRIVE_MODEL_IDS` au début du fichier `app.py`.")
    st.stop()

# Disposition en 2 colonnes
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
            st.markdown(f'<div class="badge-fail">❌ PIÈCE DÉFECTUEUSE (Score: {res.score:.3f})</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="badge-pass">✅ PIÈCE CONFORME (Score: {res.score:.3f})</div>', unsafe_allow_html=True)
            
        st.write("")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Score Anomalie</div><div class="metric-value">{res.score:.2f}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Surface Impactée</div><div class="metric-value">{100 * res.mask.mean():.1f}%</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Latence</div><div class="metric-value">{res.latency_ms:.0f} ms</div></div>', unsafe_allow_html=True)
            
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
