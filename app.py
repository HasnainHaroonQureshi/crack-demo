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

# NOTE: IoU slider kept for model NMS control (not displayed as metric)
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
# DOWNLOAD MODEL FROM GITHUB
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

    max_radius = dist.max()
    max_width = max_radius * 2

    max_loc = np.unravel_index(np.argmax(dist), dist.shape)
    y, x = max_loc

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

    combined_mask = np.zeros(masks[0].shape, dtype=np.uint8)

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
def draw_combined_results(img, res_coin, crack_mask, width_mm, max_width_point):
    output = img.copy()

    if res_coin.boxes is not None and len(res_coin.boxes) > 0:
        for box in res_coin.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)

    if crack_mask is not None:
        red_mask = np.zeros_like(output)
        red_mask[:, :, 2] = crack_mask
        output = cv2.addWeighted(output, 1.0, red_mask, 0.4, 0)

    if max_width_point is not None:
        x, y = max_width_point
        cv2.circle(output, (x, y), 10, (0, 255, 0), -1)

    if width_mm is not None:
        cv2.putText(
            output,
            f"Width: {width_mm:.2f} mm",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 0),
            3,
        )

    return output


# =========================================================
# MAIN UI
# =========================================================
st.title("🏗️ StructInsight AI")

uploaded_file = st.file_uploader("Upload Concrete Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(img_np, caption="Uploaded Image", width="stretch")

    try:
        with st.status("Loading AI Models..."):
            crack_model = get_model("crack_model.pt", CRACK_MODEL_PATH)
            coin_model = get_model("coin_model.pt", COIN_MODEL_PATH)

        res_coin = run_yolo_predict(coin_model, img_np, confidence_threshold, iou_threshold)
        res_crack = run_yolo_predict(crack_model, img_np, confidence_threshold, iou_threshold)

        # COIN
        coin_px = None
        if res_coin.boxes is not None and len(res_coin.boxes) > 0:
            boxes = res_coin.boxes.xyxy.cpu().numpy()
            b = max(boxes, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))
            x1, y1, x2, y2 = map(int, b)
            coin_px = ((x2-x1) + (y2-y1)) / 2

        # CRACK
        combined_mask = create_clean_mask(res_crack)
        crack_px = None
        max_width_point = None

        if combined_mask is not None:
            crack_px, max_width_point = mask_max_width_pixels(combined_mask)

        # CALIBRATION
        width_mm = None
        mm_per_pixel = None

        if use_manual_calibration:
            mm_per_pixel = manual_mm_per_pixel
        elif coin_px is not None:
            mm_per_pixel = coin_diameter_mm / coin_px

        if mm_per_pixel and crack_px:
            width_mm = crack_px * mm_per_pixel * calibration_factor

        # OUTPUT IMAGE
        result_image = draw_combined_results(
            img_np, res_coin, combined_mask, width_mm, max_width_point
        )

        st.image(result_image, caption="Detection Result", width="stretch")

        # =================================================
        # METRICS (NO IOU)
        # =================================================

        avg_conf = None
        conf_list = []

        if res_crack.boxes is not None and len(res_crack.boxes) > 0:
            conf_list.extend(res_crack.boxes.conf.cpu().numpy())

        if res_coin.boxes is not None and len(res_coin.boxes) > 0:
            conf_list.extend(res_coin.boxes.conf.cpu().numpy())

        if conf_list:
            avg_conf = float(np.mean(conf_list))

        seg_score = None
        if combined_mask is not None:
            seg_score = np.sum(combined_mask > 0) / combined_mask.size

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Model Confidence", f"{avg_conf:.2f}" if avg_conf else "N/A")

        with c2:
            st.metric("Crack Coverage", f"{seg_score:.4f}" if seg_score else "N/A")

        with c3:
            st.metric("Calibration", f"{mm_per_pixel:.4f}" if mm_per_pixel else "N/A")

        # WIDTH
        if width_mm:
            st.success(f"Estimated Crack Width: {width_mm:.2f} mm")

    except Exception as e:
        st.error(str(e))
