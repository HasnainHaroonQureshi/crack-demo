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
GITHUB_TOKEN = safe_secret_get("github_token")
PRIVATE_REPO = safe_secret_get("private_repo")

CRACK_MODEL_PATH = safe_secret_get("crack_model_path", "Crackdetection_model.pt")
COIN_MODEL_PATH = safe_secret_get("coin_model_path", "Coindetection_model.pt")

DEFAULT_COIN_DIAMETER_MM = float(safe_secret_get("coin_diameter_mm", 18.5))

CACHE_DIR = Path("models_cache")
CACHE_DIR.mkdir(exist_ok=True)


# =========================================================
# SIDEBAR SETTINGS
# =========================================================
st.sidebar.title("⚙️ Detection Settings")

coin_diameter_mm = st.sidebar.number_input(
    "Coin Diameter (mm)",
    min_value=1.0,
    max_value=100.0,
    value=DEFAULT_COIN_DIAMETER_MM,
    step=0.1,
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=1.0,
    value=0.25,
    step=0.05,
)

iou_threshold = st.sidebar.slider(
    "IoU Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.45,
    step=0.05,
)

st.sidebar.title("🧪 Width Calibration")

calibration_factor = st.sidebar.slider(
    "Calibration Factor",
    min_value=0.10,
    max_value=3.00,
    value=1.00,
    step=0.01,
)

st.sidebar.caption("Final Width = Measured Width × Calibration Factor")

st.sidebar.title("📏 Calibration")

use_manual_calibration = st.sidebar.checkbox(
    "Use Manual Calibration",
    value=False,
)

manual_mm_per_pixel = st.sidebar.number_input(
    "Manual mm/pixel Calibration",
    min_value=0.0001,
    max_value=10.0,
    value=0.05,
    step=0.001,
    format="%.4f",
)


# =========================================================
# DOWNLOAD MODEL
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
# LOAD MODEL
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
        raise RuntimeError(f"Model not found: {repo_path}")

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

    max_loc = np.unravel_index(np.argmax(dist), dist.shape)
    y, x = max_loc

    return dist.max() * 2, (x, y)


# =========================================================
# CLEAN MASK
# =========================================================
def create_clean_mask(res_crack):
    if res_crack.masks is None:
        return None

    masks = res_crack.masks.data.cpu().numpy()
    if len(masks) == 0:
        return None

    combined = np.zeros((masks[0].shape[0], masks[0].shape[1]), dtype=np.uint8)

    for mask in masks:
        binary = (mask > 0.5).astype(np.uint8) * 255
        combined = cv2.bitwise_or(combined, binary)

    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.medianBlur(combined, 5)

    return combined


# =========================================================
# DRAW RESULTS
# =========================================================
def draw_combined_results(img, res_coin, crack_mask, width_mm=None, max_width_point=None):
    output = img.copy()

    if res_coin.boxes is not None and len(res_coin.boxes) > 0:
        for box in res_coin.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)

    if crack_mask is not None:
        red = np.zeros_like(output)
        red[:, :, 2] = crack_mask
        output = cv2.addWeighted(output, 1.0, red, 0.4, 0)

    if max_width_point is not None:
        x, y = max_width_point
        cv2.circle(output, (x, y), 8, (0, 255, 0), -1)

    if width_mm is not None:
        cv2.putText(
            output,
            f"{width_mm:.2f} mm",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 0),
            3,
        )

    return output


# =========================================================
# APP
# =========================================================
st.title("🏗️ StructInsight AI")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(img_np, caption="Input Image", width="stretch")

    try:
        with st.status("Loading models..."):
            crack_model = get_model("crack_model.pt", CRACK_MODEL_PATH)
            coin_model = get_model("coin_model.pt", COIN_MODEL_PATH)

        res_coin = run_yolo_predict(coin_model, img_np, confidence_threshold, iou_threshold)
        res_crack = run_yolo_predict(crack_model, img_np, confidence_threshold, iou_threshold)

        # =====================================================
        # MAX CONFIDENCE (REAL VALUE)
        # =====================================================
        max_conf = None
        if res_crack.boxes is not None and len(res_crack.boxes) > 0:
            max_conf = float(np.max(res_crack.boxes.conf.cpu().numpy()))

        coin_px = None
        if res_coin.boxes is not None and len(res_coin.boxes) > 0:
            boxes = res_coin.boxes.xyxy.cpu().numpy()
            b = max(boxes, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))
            x1, y1, x2, y2 = map(int, b)
            coin_px = ((x2-x1)+(y2-y1))/2

        mask = create_clean_mask(res_crack)
        crack_px = None
        max_pt = None

        if mask is not None:
            crack_px, max_pt = mask_max_width_pixels(mask)

        mm_per_pixel = None
        if use_manual_calibration:
            mm_per_pixel = manual_mm_per_pixel
        elif coin_px:
            mm_per_pixel = coin_diameter_mm / coin_px

        width_mm = None
        if mm_per_pixel and crack_px:
            width_mm = crack_px * mm_per_pixel * calibration_factor

        result = draw_combined_results(img_np, res_coin, mask, width_mm, max_pt)

        st.subheader("Result")
        st.image(result, width="stretch")

        # =====================================================
        # METRICS
        # =====================================================
        c1, c2 = st.columns(2)

        with c1:
            st.metric("Model Confidence", f"{max_conf:.2f}" if max_conf else "N/A")

        with c2:
            st.metric("Calibration", f"{mm_per_pixel:.4f} mm/pixel" if mm_per_pixel else "N/A")

        c3, c4 = st.columns(2)

        with c3:
            st.metric("Calibration Factor", f"{calibration_factor:.2f}")

        with c4:
            st.metric("Crack Width", f"{width_mm:.2f} mm" if width_mm else "N/A")

    except Exception as e:
        st.error(f"Error: {e}")



# =========================================================
# ABOUT SECTION
# =========================================================
st.markdown("---")
st.header("📘 About the Project")
st.markdown(
    """
### Project Overview

This project focuses on the development of an advanced deep learning system for automated crack detection and severity classification using **YOLOv11**. Traditional structural assessments require manual visual inspections, which can be time-consuming, subjective, and difficult to scale across large infrastructures.

By leveraging state-of-the-art computer vision algorithms, **StructInsight AI** achieves real-time detection capabilities with high precision. The system not only identifies cracks in concrete but also provides actionable insights through severity classification.

This allows engineers and maintenance teams to:

- Quickly assess structural integrity
- Prioritize repair work
- Reduce maintenance costs
- Improve inspection efficiency
- Enhance public safety

This research highlights the critical importance of integrating AI technologies into civil engineering, showcasing how emerging tools can transform and modernize traditional infrastructure monitoring methods.
"""
)

st.markdown("---")
st.header("👨‍🔬 Research Team")
st.markdown(
    """
### Project Lead
**Hasnain Haroon**

Developed as a Final Year Project (FYP) within the Department of Civil Engineering at COMSATS University Islamabad, Wah Campus.

---

### Project Supervisor
**Engr. Sandeerah Choudhary**

---

### Key Team Members
- **Liza Liaqat**
- **Ammar Faheem Khawaja**
"""
)
