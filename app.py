import gc
import os
from pathlib import Path
import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="StructInsight AI",
    page_icon="🏗️",
    layout="wide",
)

# =========================================================
# CUSTOM CSS — premium dark engineering aesthetic
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Reset & base ─────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #080c18;
    color: #e2e8f0;
}

/* ── Hide default Streamlit chrome ───────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 1280px; }

/* ── Global scrollbar ─────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }

/* ═══════════════════════════════════════════════════════
   HERO BANNER
═══════════════════════════════════════════════════════ */
.si-hero {
    background: #070b16;
    background-image:
        radial-gradient(ellipse 100% 80% at 15% 50%, rgba(56,189,248,0.12) 0%, transparent 55%),
        radial-gradient(ellipse 70% 90% at 85% 30%, rgba(251,191,36,0.07) 0%, transparent 55%),
        radial-gradient(ellipse 50% 50% at 50% 100%, rgba(99,102,241,0.08) 0%, transparent 50%);
    padding: 3.5rem 3.5rem 3rem;
    margin: -1rem -1rem 2.5rem;
    position: relative;
    overflow: hidden;
    border-bottom: 1px solid rgba(56,189,248,0.12);
}

/* animated blueprint grid */
.si-hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        repeating-linear-gradient(90deg,  rgba(56,189,248,0.03) 0, rgba(56,189,248,0.03) 1px, transparent 1px, transparent 64px),
        repeating-linear-gradient(180deg, rgba(56,189,248,0.03) 0, rgba(56,189,248,0.03) 1px, transparent 1px, transparent 64px);
    animation: gridSlide 20s linear infinite;
}
@keyframes gridSlide {
    0%   { transform: translate(0,0); }
    100% { transform: translate(64px, 64px); }
}

/* glowing orb top-right */
.si-hero::after {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(56,189,248,0.09) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}

.si-hero-inner { position: relative; z-index: 1; }

.si-hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(251,191,36,0.1);
    border: 1px solid rgba(251,191,36,0.3);
    color: #fbbf24;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    margin-bottom: 1.25rem;
}
.si-hero-tag::before {
    content: '';
    width: 6px; height: 6px;
    background: #fbbf24;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.7); }
}

.si-hero h1 {
    color: #f1f5f9;
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: -2px;
    margin: 0 0 0.7rem;
    line-height: 1.05;
    text-shadow: 0 0 60px rgba(56,189,248,0.15);
}
.si-hero h1 span {
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.si-hero p {
    color: #64748b;
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
    max-width: 580px;
    line-height: 1.75;
}

.si-hero-badges {
    display: flex;
    gap: 0.55rem;
    margin-top: 1.75rem;
    flex-wrap: wrap;
}
.si-badge {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    color: #94a3b8;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    padding: 0.35rem 0.85rem;
    border-radius: 6px;
    transition: all 0.2s;
}
.si-badge:hover {
    background: rgba(56,189,248,0.08);
    border-color: rgba(56,189,248,0.25);
    color: #38bdf8;
}

/* ═══════════════════════════════════════════════════════
   SECTION HEADERS
═══════════════════════════════════════════════════════ */
.si-section {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 2.5rem 0 1.25rem;
}
.si-section-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(56,189,248,0.25), transparent);
}
.si-section-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #38bdf8;
    white-space: nowrap;
}
.si-section-icon {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem;
    flex-shrink: 0;
    box-shadow: 0 0 12px rgba(56,189,248,0.1);
}

/* ═══════════════════════════════════════════════════════
   UPLOAD ZONE
═══════════════════════════════════════════════════════ */
[data-testid="stFileUploader"] {
    background: rgba(15,23,42,0.8);
    border: 2px dashed rgba(56,189,248,0.25) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    transition: all 0.3s !important;
    box-shadow: inset 0 0 40px rgba(56,189,248,0.03);
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(56,189,248,0.55) !important;
    background: rgba(56,189,248,0.04) !important;
    box-shadow: inset 0 0 40px rgba(56,189,248,0.06), 0 0 20px rgba(56,189,248,0.06);
}
[data-testid="stFileUploaderDropzoneInstructions"] { color: #475569 !important; }
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #0369a1, #0284c7) !important;
    border: none !important;
    color: #fff !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.04em !important;
}

/* ═══════════════════════════════════════════════════════
   METRIC CARDS
═══════════════════════════════════════════════════════ */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #0d1421, #111827);
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    border-bottom: 3px solid #38bdf8;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), 0 0 0 1px rgba(56,189,248,0.05);
    transition: transform 0.2s, box-shadow 0.2s;
    position: relative;
    overflow: hidden;
}
[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.3), transparent);
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.5), 0 0 20px rgba(56,189,248,0.08);
}
[data-testid="stMetricLabel"] {
    color: #475569 !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
}
[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    font-variant-numeric: tabular-nums !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ═══════════════════════════════════════════════════════
   IMAGE PANELS
═══════════════════════════════════════════════════════ */
.si-img-panel {
    background: #0d1421;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 1.25rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: border-color 0.3s;
}
.si-img-panel:hover { border-color: rgba(56,189,248,0.25); }
.si-img-label {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.si-img-label::before {
    content: '';
    display: inline-block;
    width: 3px; height: 12px;
    background: #38bdf8;
    border-radius: 2px;
}
[data-testid="stImage"] img {
    border-radius: 10px;
}

/* ═══════════════════════════════════════════════════════
   SEVERITY CARD
═══════════════════════════════════════════════════════ */
.si-severity-wrap {
    background: linear-gradient(145deg, #0d1421, #111827);
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 1.75rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.si-severity-wrap::after {
    content: '';
    position: absolute;
    bottom: 0; right: 0;
    width: 120px; height: 120px;
    background: radial-gradient(circle, rgba(56,189,248,0.05) 0%, transparent 70%);
    pointer-events: none;
}
.si-sev-header {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.6rem;
}
.si-sev-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 1.1rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    font-family: 'JetBrains Mono', monospace;
}
.si-sev-value {
    font-size: 2.8rem;
    font-weight: 900;
    font-variant-numeric: tabular-nums;
    line-height: 1;
    margin-bottom: 0.25rem;
    font-family: 'JetBrains Mono', monospace;
}
.si-sev-unit { font-size: 1.1rem; font-weight: 500; color: #475569; }
.si-info-block {
    background: rgba(15,23,42,0.6);
    border: 1px solid #1e293b;
    border-left: 3px solid #1e293b;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    font-size: 0.875rem;
    color: #94a3b8;
    line-height: 1.7;
    transition: border-left-color 0.2s;
}
.si-info-block strong {
    color: #cbd5e1;
    display: block;
    margin-bottom: 0.3rem;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 700;
}

/* ═══════════════════════════════════════════════════════
   ABOUT & TEAM CARDS
═══════════════════════════════════════════════════════ */
.si-card {
    background: linear-gradient(145deg, #0d1421, #111827);
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 1.75rem;
    height: 100%;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    margin-bottom: 1rem;
    transition: border-color 0.3s, transform 0.2s;
    position: relative;
    overflow: hidden;
}
.si-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.2), transparent);
}
.si-card:hover {
    border-color: rgba(56,189,248,0.2);
    transform: translateY(-2px);
}
.si-card-tag {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 0.5rem;
}
.si-card h3 { margin: 0 0 0.85rem; color: #f1f5f9; font-size: 1.05rem; font-weight: 700; }
.si-card p  { margin: 0; color: #64748b; font-size: 0.875rem; line-height: 1.75; }
.si-cap-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.si-cap {
    background: rgba(3,105,161,0.15);
    border: 1px solid rgba(56,189,248,0.2);
    color: #38bdf8;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 0.25rem 0.7rem;
    border-radius: 6px;
    letter-spacing: 0.04em;
}
.si-ref-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.65rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    transition: background 0.2s;
}
.si-ref-row:last-child { border-bottom: none; }
.si-ref-row:hover { background: rgba(255,255,255,0.02); border-radius: 6px; padding-left: 4px; }
.si-ref-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; box-shadow: 0 0 6px currentColor; }
.si-ref-range { font-size: 0.75rem; color: #475569; margin-left: auto; font-variant-numeric: tabular-nums; font-family: 'JetBrains Mono', monospace; }
.si-ref-label { font-size: 0.82rem; font-weight: 600; color: #e2e8f0; }
.si-ref-action { font-size: 0.72rem; color: #64748b; }

.si-team-role { font-size: 0.62rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase; color: #38bdf8; margin-bottom: 0.35rem; }
.si-team-name { font-size: 1.05rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.4rem; }
.si-team-detail { font-size: 0.78rem; color: #64748b; line-height: 1.6; }
.si-team-divider { width: 28px; height: 2px; background: linear-gradient(90deg, #38bdf8, #818cf8); margin: 0.7rem 0; border-radius: 2px; }

/* ═══════════════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #060a14 !important;
    border-right: 1px solid #0f1a2e !important;
}
[data-testid="stSidebarContent"] { padding-top: 0 !important; }

/* Sidebar logo strip */
[data-testid="stSidebar"] > div > div > div > div:first-child::before {
    content: '🏗️  STRUCTINSIGHT';
    display: block;
    font-size: 0.6rem;
    font-weight: 800;
    letter-spacing: 0.2em;
    color: #38bdf8;
    padding: 1.5rem 1rem 1rem;
    border-bottom: 1px solid #0f1a2e;
    margin-bottom: 0.5rem;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSlider span,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stCheckbox label {
    color: #64748b !important;
    font-size: 0.78rem !important;
}
[data-testid="stSidebar"] h1 {
    color: #1e293b !important;
    font-size: 0.58rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    padding: 1.25rem 0 0.5rem !important;
    border-bottom: 1px solid #0f1a2e !important;
    margin-bottom: 0.85rem !important;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}

/* Sidebar number inputs & sliders */
[data-testid="stSidebar"] [data-baseweb="input"] {
    background: #0d1421 !important;
    border-color: #1e293b !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] div[role="slider"] {
    background: #38bdf8 !important;
    box-shadow: 0 0 10px rgba(56,189,248,0.5) !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div:first-child {
    background: #1e293b !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div:nth-child(2) {
    background: linear-gradient(90deg, #38bdf8, #818cf8) !important;
}
[data-testid="stSidebar"] .stCaption { color: #1e293b !important; font-size: 0.7rem !important; }

/* ═══════════════════════════════════════════════════════
   STATUS / ALERTS
═══════════════════════════════════════════════════════ */
[data-testid="stStatusWidget"] {
    background: #0d1421 !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
}
.si-error {
    background: rgba(220,38,38,0.08);
    border: 1px solid rgba(220,38,38,0.25);
    border-left: 3px solid #dc2626;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    color: #fca5a5;
    font-size: 0.875rem;
}

/* Streamlit alerts */
[data-testid="stAlert"] {
    background: rgba(15,23,42,0.8) !important;
    border-radius: 10px !important;
    border: 1px solid #1e293b !important;
}

/* ═══════════════════════════════════════════════════════
   DIVIDER & FOOTER
═══════════════════════════════════════════════════════ */
hr { border: none; border-top: 1px solid #0f1a2e; margin: 2.5rem 0; }

.si-footer {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    color: #1e293b;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    border-top: 1px solid #0f1a2e;
    margin-top: 1rem;
}
.si-footer span { color: #38bdf8; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
def safe_secret_get(key, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

# =========================================================
# CONFIG
# =========================================================
GITHUB_TOKEN           = safe_secret_get("github_token")
PRIVATE_REPO           = safe_secret_get("private_repo")
CRACK_MODEL_PATH       = safe_secret_get("crack_model_path", "Crackdetection_model.pt")
COIN_MODEL_PATH        = safe_secret_get("coin_model_path",  "Coindetection_model.pt")
DEFAULT_COIN_DIAMETER_MM = float(safe_secret_get("coin_diameter_mm", 18.5))
CACHE_DIR = Path("models_cache")
CACHE_DIR.mkdir(exist_ok=True)

# =========================================================
# SIDEBAR SETTINGS
# =========================================================
st.sidebar.title("Detection Settings")
coin_diameter_mm = st.sidebar.number_input(
    "Coin Diameter (mm)",
    min_value=1.0, max_value=100.0,
    value=DEFAULT_COIN_DIAMETER_MM, step=0.1,
)
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05, max_value=1.0, value=0.25, step=0.05,
)
iou_threshold = st.sidebar.slider(
    "IoU Threshold",
    min_value=0.1, max_value=1.0, value=0.45, step=0.05,
)

st.sidebar.title("Width Calibration")
calibration_factor = st.sidebar.slider(
    "Calibration Factor",
    min_value=0.10, max_value=3.00, value=1.00, step=0.01,
)
st.sidebar.caption("Final Width = Measured Width × Calibration Factor")

st.sidebar.title("Scale Calibration")
use_manual_calibration = st.sidebar.checkbox("Use Manual Calibration", value=False)
manual_mm_per_pixel = st.sidebar.number_input(
    "Manual mm/pixel",
    min_value=0.0001, max_value=10.0,
    value=0.05, step=0.001, format="%.4f",
)

# =========================================================
# DOWNLOAD MODEL FROM PRIVATE GITHUB
# =========================================================
def download_from_github(repo, filepath, dest_path):
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    for branch in ("main", "master"):
        url = f"https://raw.githubusercontent.com/{repo}/{branch}/{filepath}"
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        if response.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return dest_path
    raise RuntimeError(f"Download failed for {filepath}")

# =========================================================
# LOAD YOLO MODELS
# =========================================================
@st.cache_resource
def get_model(local_name, repo_path):
    local_file = CACHE_DIR / local_name
    if local_file.exists():
        return YOLO(str(local_file))
    source_path = Path(repo_path)
    if source_path.exists():
        return YOLO(str(source_path))
    if not PRIVATE_REPO:
        raise RuntimeError(
            f"Model not found locally and PRIVATE_REPO is not set: {repo_path}"
        )
    download_from_github(PRIVATE_REPO, repo_path, local_file)
    return YOLO(str(local_file))

# =========================================================
# YOLO PREDICTION
# =========================================================
def _free_memory():
    """Force-clear unreferenced tensors/arrays between inference runs."""
    gc.collect()

def _resize_for_inference(img_np, max_dim=1024):
    """Downscale longest side to max_dim before inference to cap RAM usage."""
    h, w = img_np.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_np

def run_yolo_predict(model, img, conf_threshold, iou_thresh):
    # Shrink image first so each model run uses ~4x less RAM
    small   = _resize_for_inference(img, max_dim=1024)
    pil_img = Image.fromarray(small)
    results = model.predict(
        source=pil_img,
        imgsz=1024,
        conf=conf_threshold,
        iou=iou_thresh,
        retina_masks=True,
        verbose=False,
    )
    result = results[0]
    # Pull everything to CPU and delete the results list immediately
    if result.boxes is not None:
        result.boxes = result.boxes.cpu()
    if result.masks is not None:
        result.masks = result.masks.cpu()
    del results
    _free_memory()
    return result

# =========================================================
# WIDTH CALCULATION — perpendicular cross-section method
# =========================================================
def mask_max_width_pixels(mask):
    """
    Returns (max_width_px, (x, y)) where max_width_px is the widest
    perpendicular cross-section of the crack mask, measured by:
      1. Skeletonising the mask to find the crack centreline.
      2. At each skeleton pixel, casting a perpendicular ray in both
         directions until it exits the mask.
      3. Returning the longest such measurement and the pixel where it occurs.
    Falls back to the distance-transform estimate if skeleton fails.
    """
    binary = (mask > 0).astype(np.uint8)

    # -- skeletonise (Zhang-Suen thinning via morphology) --------------------
    skel   = np.zeros_like(binary)
    temp   = binary.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded   = cv2.erode(temp, kernel)
        opened   = cv2.dilate(eroded, kernel)
        skel     = cv2.bitwise_or(skel, cv2.subtract(temp, opened))
        temp     = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    skel_pts = np.column_stack(np.where(skel > 0))   # (row, col) pairs

    if len(skel_pts) < 2:
        # fallback: distance transform
        dist      = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        max_r     = float(dist.max())
        loc       = np.unravel_index(np.argmax(dist), dist.shape)
        return max_r * 2.0, (int(loc[1]), int(loc[0]))

    # -- estimate local orientation via PCA on skeleton points ---------------
    skel_f   = skel_pts.astype(np.float32)
    mean_pt  = skel_f.mean(axis=0)
    _, _, vt = np.linalg.svd(skel_f - mean_pt, full_matrices=False)
    tang     = vt[0]                          # principal direction (tangent)
    perp     = np.array([-tang[1], tang[0]])  # perpendicular direction

    h, w     = binary.shape
    max_width = 0.0
    best_pt   = (int(mean_pt[1]), int(mean_pt[0]))

    # sample every 4th skeleton pixel for speed
    for pt in skel_pts[::4]:
        r, c = pt
        # cast ray in +perp and -perp directions
        width = 0.0
        for sign in (1, -1):
            for step in range(1, max(h, w)):
                nr = int(round(r + sign * step * perp[0]))
                nc = int(round(c + sign * step * perp[1]))
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    break
                if binary[nr, nc] == 0:
                    break
                width += 1.0
        if width > max_width:
            max_width = width
            best_pt   = (c, r)   # (x, y) for cv2

    return max_width, best_pt

# =========================================================
# CLEAN MASK
# =========================================================
def create_clean_mask(res_crack, orig_h, orig_w):
    if res_crack.masks is None:
        return None
    masks = res_crack.masks.data.cpu().numpy()
    if len(masks) == 0:
        return None
    combined_mask = np.zeros(
        (masks[0].shape[0], masks[0].shape[1]), dtype=np.uint8
    )
    for mask in masks:
        binary = (mask > 0.5).astype(np.uint8) * 255
        combined_mask = cv2.bitwise_or(combined_mask, binary)
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.medianBlur(combined_mask, 5)
    # Resize mask to match the original image so overlays align correctly
    if combined_mask.shape[0] != orig_h or combined_mask.shape[1] != orig_w:
        combined_mask = cv2.resize(
            combined_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
    return combined_mask

# =========================================================
# DRAW RESULTS
# =========================================================
def draw_combined_results(img, coin_boxes, crack_mask, width_mm=None, max_width_point=None):
    output = img.copy()
    h, w   = output.shape[:2]
    thick  = max(2, h // 400)   # scale line thickness to image size
    font_s = max(0.6, h / 1200)

    # ── Crack mask overlay (semi-transparent red) ──────────────────────────
    if crack_mask is not None:
        red_layer          = output.copy()
        crack_bool         = crack_mask > 0
        red_layer[crack_bool] = [220, 50, 50]   # BGR red
        output = cv2.addWeighted(output, 0.65, red_layer, 0.35, 0)

        # Draw crack contours for crisp edges
        contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (255, 80, 80), thick)

    # ── Coin detection overlay ─────────────────────────────────────────────
    if coin_boxes:
        for box in coin_boxes:
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            rx, ry = (x2 - x1) // 2, (y2 - y1) // 2
            radius = (rx + ry) // 2

            # Semi-transparent green fill
            overlay = output.copy()
            cv2.circle(overlay, (cx, cy), radius, (0, 220, 100), -1)
            output = cv2.addWeighted(output, 0.85, overlay, 0.15, 0)

            # Circle outline
            cv2.circle(output, (cx, cy), radius, (0, 220, 100), thick + 1)
            # Cross-hair centre
            cv2.line(output, (cx - 12, cy), (cx + 12, cy), (0, 220, 100), thick)
            cv2.line(output, (cx, cy - 12), (cx, cy + 12), (0, 220, 100), thick)

            # Label with background
            label     = "REF COIN"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_s * 0.75, thick)
            lx, ly    = x1, max(y1 - 10, lh + 6)
            cv2.rectangle(output, (lx, ly - lh - 4), (lx + lw + 8, ly + 4), (0, 220, 100), -1)
            cv2.putText(output, label, (lx + 4, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, font_s * 0.75, (0, 0, 0), thick)

    # ── Max-width point ────────────────────────────────────────────────────
    if max_width_point is not None:
        x, y = max_width_point
        cv2.circle(output, (x, y), 14, (255, 220, 0), -1)
        cv2.circle(output, (x, y), 14, (255, 255, 255), 2)
        label     = "MAX WIDTH"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_s * 0.75, thick)
        lx = min(x + 18, w - lw - 10)
        ly = max(y - 10, lh + 6)
        cv2.rectangle(output, (lx - 2, ly - lh - 4), (lx + lw + 6, ly + 4), (40, 40, 40), -1)
        cv2.putText(output, label, (lx + 2, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, font_s * 0.75, (255, 220, 0), thick)

    # ── Width annotation (top-left panel) ─────────────────────────────────
    if width_mm is not None:
        text      = f"Crack Width: {width_mm:.3f} mm"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_s, thick + 1)
        pad       = 12
        cv2.rectangle(output, (20, 20), (20 + tw + pad * 2, 20 + th + pad * 2), (15, 15, 15), -1)
        cv2.rectangle(output, (20, 20), (20 + tw + pad * 2, 20 + th + pad * 2), (255, 220, 0), 2)
        cv2.putText(output, text, (20 + pad, 20 + pad + th),
                    cv2.FONT_HERSHEY_SIMPLEX, font_s, (255, 220, 0), thick + 1)

    return output

# =========================================================
# PAGE HEADER
# =========================================================
st.markdown("""
<div class="si-hero">
    <div class="si-hero-inner">
        <div class="si-hero-tag">⚡ AI-Powered Structural Analysis</div>
        <h1>Struct<span>Insight</span> AI</h1>
        <p>Automated concrete crack detection &amp; severity assessment using YOLOv11 instance segmentation. Upload an image with a reference coin for precise real-world measurements.</p>
        <div class="si-hero-badges">
            <span class="si-badge">🔬 YOLOv11 Segmentation</span>
            <span class="si-badge">📐 Coin Calibration</span>
            <span class="si-badge">📊 ACI 224R Severity Scale</span>
            <span class="si-badge">🏛 COMSATS University Wah</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# IMAGE UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
    "Upload a concrete surface image",
    type=["jpg", "jpeg", "png"],
)

# =========================================================
# AUTO CLEANUP: wipe previous run's data when a new image arrives
# =========================================================
current_id = id(uploaded_file) if uploaded_file is not None else None

if "last_file_id" not in st.session_state:
    st.session_state.last_file_id = None

if current_id != st.session_state.last_file_id:
    # New image detected — clear everything from the previous run
    for key in ["result_image", "combined_mask", "coin_boxes",
                "avg_conf", "mask_coverage", "width_mm",
                "mm_per_pixel", "coin_px", "crack_px", "max_width_point"]:
        if key in st.session_state:
            del st.session_state[key]
    _free_memory()
    st.session_state.last_file_id = current_id

# =========================================================
# PROCESS IMAGE
# =========================================================
if uploaded_file:
    image  = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.markdown('''
    <div class="si-section">
        <div class="si-section-icon">📷</div>
        <span class="si-section-label">Input Image</span>
        <div class="si-section-line"></div>
    </div>''', unsafe_allow_html=True)
    st.markdown('<div class="si-img-panel"><div class="si-img-label">Uploaded Image</div>', unsafe_allow_html=True)
    st.image(img_np, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

    try:
        # Load models (cached — only loads once ever)
        with st.status("Loading AI models…") as status:
            crack_model = get_model("crack_model.pt", CRACK_MODEL_PATH)
            coin_model  = get_model("coin_model.pt",  COIN_MODEL_PATH)
            status.update(label="Models loaded successfully.", state="complete")

        # --- Coin model ---
        try:
            res_coin = run_yolo_predict(coin_model, img_np, confidence_threshold, iou_threshold)
        except Exception as e:
            st.warning(f"Coin detection failed: {e}")
            res_coin = None

        # Extract what we need from coin result immediately, then free it
        coin_px    = None
        coin_boxes = []   # scaled to original image size for drawing
        orig_h, orig_w = img_np.shape[:2]
        if res_coin is not None and res_coin.boxes is not None and len(res_coin.boxes) > 0:
            boxes_raw = res_coin.boxes.xyxy.cpu().numpy()  # in inference (1024-scaled) coords

            # Scale factor: inference image was resized so longest side = 1024
            infer_longest = 1024.0
            orig_longest  = float(max(orig_h, orig_w))
            scale_back    = orig_longest / infer_longest

            boxes_orig = boxes_raw * scale_back   # back to original pixel space
            coin_boxes = boxes_orig.tolist()

            largest    = max(boxes_orig, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            x1, y1, x2, y2 = map(int, largest)
            # coin_px measured in original-image pixels (consistent with mask which is also orig size)
            coin_px    = ((x2 - x1) + (y2 - y1)) / 2.0
        del res_coin
        _free_memory()   # free coin tensors before crack model runs

        # --- Crack model ---
        try:
            res_crack = run_yolo_predict(crack_model, img_np, confidence_threshold, iou_threshold)
        except Exception as e:
            st.error(f"Crack detection failed: {e}")
            st.stop()

        # -------------------------------------------------
        # CRACK SEGMENTATION → extract everything to numpy, then free
        # -------------------------------------------------
        combined_mask   = create_clean_mask(res_crack, orig_h, orig_w)
        avg_conf = None
        if res_crack.boxes is not None and len(res_crack.boxes) > 0:
            avg_conf = float(res_crack.boxes.conf.cpu().numpy().mean())
        del res_crack
        _free_memory()

        crack_px        = None
        max_width_point = None
        if combined_mask is not None:
            crack_px, max_width_point = mask_max_width_pixels(combined_mask)

        # -------------------------------------------------
        # CALIBRATION
        # -------------------------------------------------
        width_mm    = None
        mm_per_pixel = None
        if use_manual_calibration:
            mm_per_pixel = manual_mm_per_pixel
        elif coin_px is not None:
            mm_per_pixel = coin_diameter_mm / coin_px

        if mm_per_pixel is not None and crack_px is not None:
            width_mm = crack_px * mm_per_pixel * calibration_factor

        # -------------------------------------------------
        # ANNOTATED IMAGE
        # -------------------------------------------------
        result_image = draw_combined_results(
            img_np, coin_boxes, combined_mask, width_mm, max_width_point
        )
        st.markdown('''
        <div class="si-section">
            <div class="si-section-icon">🔍</div>
            <span class="si-section-label">Detection Output</span>
            <div class="si-section-line"></div>
        </div>''', unsafe_allow_html=True)

        col_img_l, col_img_r = st.columns(2)
        with col_img_l:
            st.markdown('<div class="si-img-panel"><div class="si-img-label">Original</div>', unsafe_allow_html=True)
            st.image(img_np, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
        with col_img_r:
            st.markdown('<div class="si-img-panel"><div class="si-img-label">AI Detection — Cracks Highlighted</div>', unsafe_allow_html=True)
            st.image(result_image, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('''
        <div class="si-section">
            <div class="si-section-icon">📊</div>
            <span class="si-section-label">Analysis Metrics</span>
            <div class="si-section-line"></div>
        </div>''', unsafe_allow_html=True)

        # -------------------------------------------------
        # METRICS — actual model outputs, not slider values
        # avg_conf already extracted before res_crack was freed above
        # -------------------------------------------------

        mask_coverage = None
        if combined_mask is not None:
            crack_area    = np.sum(combined_mask > 0)
            total_area    = combined_mask.shape[0] * combined_mask.shape[1]
            mask_coverage = crack_area / total_area

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Detection Confidence",
                f"{avg_conf:.0%}" if avg_conf is not None else "N/A",
                help="Mean confidence score across all detected crack instances.",
            )
        with col2:
            st.metric(
                "Crack Coverage",
                f"{mask_coverage:.3%}" if mask_coverage is not None else "N/A",
                help="Percentage of image area classified as cracked.",
            )
        with col3:
            st.metric(
                "Scale",
                f"{mm_per_pixel:.4f} mm/px" if mm_per_pixel is not None else "Unavailable",
                help="Pixel-to-mm ratio derived from coin size or manual input.",
            )
        with col4:
            st.metric(
                "Max Crack Width",
                f"{width_mm:.2f} mm" if width_mm is not None else "Unavailable",
                help="Maximum crack width after applying calibration factor.",
            )

        # -------------------------------------------------
        # SEVERITY ASSESSMENT
        # -------------------------------------------------
        if width_mm is not None:
            st.markdown('''
            <div class="si-section">
                <div class="si-section-icon">🏗</div>
                <span class="si-section-label">Structural Condition Assessment</span>
                <div class="si-section-line"></div>
            </div>''', unsafe_allow_html=True)

            if width_mm < 0.1:
                severity       = "Hairline"
                badge_color    = "#16a34a"
                description    = "Barely visible cracks; typically non-structural in nature. Monitor periodically."
                recommendation = "Routine monitoring recommended. No immediate intervention required."
            elif width_mm < 0.3:
                severity       = "Minor / Fine"
                badge_color    = "#ca8a04"
                description    = (
                    "Generally acceptable in dry or interior environments. "
                    "Protective surface sealing may be required in moisture-exposed areas."
                )
                recommendation = "Consider protective sealing if the surface is exposed to moisture or weathering."
            elif width_mm <= 0.5:
                severity       = "Moderate"
                badge_color    = "#ea580c"
                description    = (
                    "Exceeds ACI 224R crack-width limits. "
                    "May indicate structural overstress or durability concerns."
                )
                recommendation = "Engineering evaluation recommended. Investigate loading and rebar condition."
            else:
                severity       = "Severe"
                badge_color    = "#dc2626"
                description    = (
                    "Significant structural distress. High risk of rebar corrosion "
                    "and progressive structural deterioration."
                )
                recommendation = "Immediate intervention required — epoxy injection, patching, or full rehabilitation."

            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"""
                <div class="si-severity-wrap" style="border-top: 4px solid {badge_color};">
                    <div class="si-sev-header">Severity Level</div>
                    <div class="si-sev-badge" style="background:{badge_color}22; color:{badge_color}; border:1px solid {badge_color}55;">
                        ● {severity}
                    </div>
                    <div class="si-sev-value" style="color:{badge_color};">{width_mm:.2f}
                        <span class="si-sev-unit">mm</span>
                    </div>
                    <div style="font-size:0.72rem; color:#94a3b8; margin-top:0.5rem;">Maximum crack width</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="si-info-block" style="border-left-color:{badge_color};">
                    <strong>Condition</strong>{description}
                </div>
                <div class="si-info-block" style="border-left-color:{badge_color};">
                    <strong>Recommended Action</strong>{recommendation}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="si-error">
                ⚠️ Width measurement unavailable — coin not detected or crack not segmented.
                Enable <strong>Manual Calibration</strong> in the sidebar to proceed without a reference coin.
            </div>''', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Processing error: {e}")

# =========================================================
# ABOUT SECTION
# =========================================================
st.markdown("""
<div class="si-section">
    <div class="si-section-icon">📘</div>
    <span class="si-section-label">About the Project</span>
    <div class="si-section-line"></div>
</div>
""", unsafe_allow_html=True)

col_about_l, col_about_r = st.columns([3, 2])
with col_about_l:
    st.markdown("""
    <div class="si-card">
        <div class="si-card-tag">Project Overview</div>
        <h3>AI-Powered Infrastructure Inspection</h3>
        <p>
        StructInsight AI applies <strong>YOLOv11 instance segmentation</strong> to automate the
        detection and severity classification of structural cracks in concrete. Traditional inspection
        methods are labour-intensive, subjective, and difficult to scale. By integrating computer vision
        into the workflow, this system delivers real-time, repeatable, and quantitative results —
        enabling engineers to prioritise repairs, reduce costs, and improve public safety.
        </p>
        <div class="si-cap-row">
            <span class="si-cap">Real-time Segmentation</span>
            <span class="si-cap">Coin Calibration</span>
            <span class="si-cap">ACI 224R Aligned</span>
            <span class="si-cap">Field-Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_about_r:
    st.markdown("""
    <div class="si-card">
        <div class="si-card-tag">Severity Reference — ACI 224R</div>
        <h3>Crack Width Classification</h3>
        <div class="si-ref-row">
            <div class="si-ref-dot" style="background:#16a34a;"></div>
            <span class="si-ref-label">Hairline</span>
            <span class="si-ref-action">Monitor only</span>
            <span class="si-ref-range">&lt; 0.1 mm</span>
        </div>
        <div class="si-ref-row">
            <div class="si-ref-dot" style="background:#ca8a04;"></div>
            <span class="si-ref-label">Minor</span>
            <span class="si-ref-action">Protective seal</span>
            <span class="si-ref-range">0.1 – 0.3 mm</span>
        </div>
        <div class="si-ref-row">
            <div class="si-ref-dot" style="background:#ea580c;"></div>
            <span class="si-ref-label">Moderate</span>
            <span class="si-ref-action">Engineering review</span>
            <span class="si-ref-range">0.3 – 0.5 mm</span>
        </div>
        <div class="si-ref-row">
            <div class="si-ref-dot" style="background:#dc2626;"></div>
            <span class="si-ref-label">Severe</span>
            <span class="si-ref-action">Immediate repair</span>
            <span class="si-ref-range">&gt; 0.5 mm</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="si-section">
    <div class="si-section-icon">👥</div>
    <span class="si-section-label">Research Team</span>
    <div class="si-section-line"></div>
</div>
""", unsafe_allow_html=True)

t1, t2, t3 = st.columns(3)
team = [
    (t1, "Project Lead", "Hasnain Haroon",
     "Final Year Project — Department of Civil Engineering, COMSATS University Islamabad, Wah Campus"),
    (t2, "Supervisor", "Engr. Sandeerah Choudhary",
     "Faculty Advisor, Department of Civil Engineering"),
    (t3, "Team Members", "Liza Liaqat · Ammar Faheem Khawaja",
     "Contributing Researchers, Civil Engineering FYP 2025"),
]
for col, role, name, detail in team:
    with col:
        st.markdown(f"""
        <div class="si-card">
            <div class="si-team-role">{role}</div>
            <div class="si-team-divider"></div>
            <div class="si-team-name">{name}</div>
            <div class="si-team-detail">{detail}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="si-footer">
    <span>StructInsight AI</span> · COMSATS University Islamabad, Wah Campus · Civil Engineering FYP 2025
</div>
""", unsafe_allow_html=True)
