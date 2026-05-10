import os
from pathlib import Path
import cv2
import numpy as np
import requests
import streamlit as st
import torch
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
# CUSTOM CSS — professional engineering aesthetic
# =========================================================
st.markdown(
    """
    <style>
    

    /* System font stack — no external network requests */
    html, body, [class*="css"] {
        font-family: ui-sans-serif, 'Segoe UI', system-ui, -apple-system, sans-serif;
    }

    /* Top header strip */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-left: 4px solid #f59e0b;
        padding: 1.5rem 2rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        color: #f8fafc;
        font-size: 1.9rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 0.85rem;
        margin: 0.4rem 0 0;
        font-weight: 300;
    }

    /* Section labels */
    h2, h3 { color: #1e293b; font-weight: 600; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        border-top: 3px solid #f59e0b;
    }
    [data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
    [data-testid="stMetricValue"] { color: #0f172a !important; font-family: ui-monospace, 'Cascadia Code', 'Fira Code', monospace !important; font-size: 1.3rem !important; }

    /* Dividers */
    hr { border-color: #e2e8f0; }

    /* Severity badge */
    .severity-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* Sidebar polish */
    [data-testid="stSidebar"] {
        background: #0f172a;
    }
    [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: #f8fafc !important; font-size: 0.85rem !important; text-transform: uppercase; letter-spacing: 0.08em; border-bottom: 1px solid #1e293b; padding-bottom: 0.4rem; }

    /* Caption / help text */
    .caption-note {
        color: #94a3b8;
        font-size: 0.78rem;
        margin-top: -0.5rem;
    }

    /* About section */
    .about-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .about-card h4 { margin: 0 0 0.25rem; color: #0f172a; font-weight: 600; }
    .about-card p  { margin: 0; color: #475569; font-size: 0.88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

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
def run_yolo_predict(model, img, conf_threshold, iou_thresh):
    results = model.predict(
        source=img,
        imgsz=1024,
        conf=conf_threshold,
        iou=iou_thresh,
        retina_masks=True,
        verbose=False,
    )
    return results[0]

# =========================================================
# WIDTH CALCULATION
# =========================================================
def mask_max_width_pixels(mask):
    mask = (mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_radius   = dist.max()
    max_width    = max_radius * 2
    max_loc      = np.unravel_index(np.argmax(dist), dist.shape)
    y, x         = max_loc
    return max_width, (x, y)

# =========================================================
# CLEAN MASK
# =========================================================
def create_clean_mask(res_crack):
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
    return combined_mask

# =========================================================
# DRAW RESULTS
# =========================================================
def draw_combined_results(img, res_coin, crack_mask, width_mm=None, max_width_point=None):
    output = img.copy()

    # Coin bounding box
    if res_coin.boxes is not None and len(res_coin.boxes) > 0:
        for box in res_coin.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(output, "Coin", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Crack mask overlay
    if crack_mask is not None:
        red_mask = np.zeros_like(output)
        red_mask[:, :, 2] = crack_mask
        output = cv2.addWeighted(output, 1.0, red_mask, 0.4, 0)

    # Max-width marker
    if max_width_point is not None:
        x, y = max_width_point
        cv2.circle(output, (x, y), 10, (0, 255, 0), -1)
        cv2.putText(output, "Max Width", (x + 15, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Width annotation
    if width_mm is not None:
        cv2.putText(output, f"Width: {width_mm:.2f} mm", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 80, 0), 3)

    return output

# =========================================================
# PAGE HEADER
# =========================================================
st.markdown(
    """
    <div class="main-header">
        <h1>🏗️ StructInsight AI</h1>
        <p>Automated structural crack detection &amp; severity assessment — powered by YOLOv11 instance segmentation</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# IMAGE UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
    "Upload a concrete surface image",
    type=["jpg", "jpeg", "png"],
)

# =========================================================
# PROCESS IMAGE
# =========================================================
if uploaded_file:
    image  = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(img_np, caption="Uploaded image", width="100%")

    try:
        # Load models
        with st.status("Loading AI models…") as status:
            crack_model = get_model("crack_model.pt", CRACK_MODEL_PATH)
            coin_model  = get_model("coin_model.pt",  COIN_MODEL_PATH)
            status.update(label="Models loaded successfully.", state="complete")

        # Run inference
        res_coin  = run_yolo_predict(coin_model,  img_np, confidence_threshold, iou_threshold)
        res_crack = run_yolo_predict(crack_model, img_np, confidence_threshold, iou_threshold)

        # -------------------------------------------------
        # COIN DETECTION → scale factor
        # -------------------------------------------------
        coin_px = None
        if res_coin.boxes is not None and len(res_coin.boxes) > 0:
            boxes      = res_coin.boxes.xyxy.cpu().numpy()
            largest    = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            x1, y1, x2, y2 = map(int, largest)
            coin_px    = ((x2 - x1) + (y2 - y1)) / 2

        # -------------------------------------------------
        # CRACK SEGMENTATION → mask + width
        # -------------------------------------------------
        combined_mask   = create_clean_mask(res_crack)
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
            img_np, res_coin, combined_mask, width_mm, max_width_point
        )
        st.subheader("Detection Output")
        st.image(result_image, caption="AI-annotated result", width="100%")

        st.markdown("---")
        st.subheader("Analysis Metrics")

        # -------------------------------------------------
        # METRICS — actual model outputs, not slider values
        # -------------------------------------------------
        avg_conf = None
        if res_crack.boxes is not None and len(res_crack.boxes) > 0:
            confidences = res_crack.boxes.conf.cpu().numpy()
            avg_conf    = float(np.mean(confidences))

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
            st.markdown("---")
            st.subheader("Structural Condition Assessment")

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
                st.markdown(
                    f"""
                    <div style="text-align:center; padding:1.5rem; background:#f8fafc;
                         border:1px solid #e2e8f0; border-radius:6px; border-top:4px solid {badge_color};">
                        <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
                             color:#64748b; margin-bottom:0.4rem;">Severity Level</div>
                        <div style="font-size:1.6rem; font-weight:700; color:{badge_color};">{severity}</div>
                        <div style="font-size:1.1rem; color:#0f172a; margin-top:0.3rem;
                             font-family:ui-monospace, 'Cascadia Code', 'Fira Code', monospace;">{width_mm:.2f} mm</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""
                    <div class="about-card">
                        <h4>Condition Description</h4>
                        <p>{description}</p>
                    </div>
                    <div class="about-card" style="border-left:3px solid {badge_color};">
                        <h4>Recommended Action</h4>
                        <p>{recommendation}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.error(
                "Width measurement unavailable. Possible causes: coin not detected, "
                "crack not segmented, or poor image quality."
            )

    except Exception as e:
        st.error(f"Processing error: {e}")

# =========================================================
# ABOUT SECTION
# =========================================================
st.markdown("---")
st.header("About the Project")

st.markdown(
    """
    <div class="about-card">
        <h4>Project Overview</h4>
        <p>
        StructInsight AI applies <strong>YOLOv11 instance segmentation</strong> to automate the
        detection and severity classification of structural cracks in concrete.
        Traditional inspection methods are labour-intensive, subjective, and difficult to
        scale across large infrastructure networks. By integrating computer vision into the
        assessment workflow, the system delivers real-time, repeatable, and quantitative results
        — enabling engineers to prioritise repairs, reduce maintenance costs, and improve
        public safety outcomes.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown(
        """
        <div class="about-card">
            <h4>Key Capabilities</h4>
            <p>
            • Real-time crack segmentation and width estimation<br>
            • Coin-based automatic pixel-to-mm scale calibration<br>
            • ACI 224R-aligned severity classification<br>
            • Configurable confidence and IoU thresholds<br>
            • Manual calibration fallback for fieldwork
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_b:
    st.markdown(
        """
        <div class="about-card">
            <h4>Severity Reference (ACI 224R)</h4>
            <p>
            <strong style="color:#16a34a;">Hairline</strong> — &lt; 0.1 mm — Monitor only<br>
            <strong style="color:#ca8a04;">Minor</strong> — 0.1–0.3 mm — Protective sealing<br>
            <strong style="color:#ea580c;">Moderate</strong> — 0.3–0.5 mm — Engineering review<br>
            <strong style="color:#dc2626;">Severe</strong> — &gt; 0.5 mm — Immediate repair
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.header("Research Team")

team_cols = st.columns(3)
team = [
    ("Project Lead",       "Hasnain Haroon",        "Final Year Project — Civil Engineering, COMSATS University Islamabad, Wah Campus"),
    ("Supervisor",         "Engr. Sandeerah Choudhary", "Faculty Advisor, Department of Civil Engineering"),
    ("Team Members",       "Liza Liaqat · Ammar Faheem Khawaja", "Contributing Researchers"),
]
for col, (role, name, detail) in zip(team_cols, team):
    with col:
        st.markdown(
            f"""
            <div class="about-card">
                <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; color:#64748b;">{role}</div>
                <h4 style="margin-top:0.25rem;">{name}</h4>
                <p>{detail}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
