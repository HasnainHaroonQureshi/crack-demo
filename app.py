import os
import requests
import numpy as np
import cv2
import torch
import streamlit as st
from PIL import Image
from pathlib import Path
from ultralytics import YOLO

# ---------- CONFIG (set via Streamlit secrets) ----------
# Ensure these names match your "Secrets" in the Streamlit Dashboard
GITHUB_TOKEN = st.secrets.get("github_token")
PRIVATE_REPO = st.secrets.get("private_repo")  
CRACK_MODEL_PATH = st.secrets.get("crack_model_path", "Crackdetection_model.pt")
COIN_MODEL_PATH  = st.secrets.get("coin_model_path", "Coindetection_model.pt")
COIN_DIAMETER_MM = float(st.secrets.get("coin_diameter_mm", "18.5"))

CACHE_DIR = Path("models_cache")
CACHE_DIR.mkdir(exist_ok=True)

# ---------- Helper: Download from private GitHub ----------
def download_from_github(repo, filepath, dest_path):
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}"
    }

    url = f"https://raw.githubusercontent.com/{repo}/main/{filepath}"

    r = requests.get(url, headers=headers, stream=True)

    if r.status_code != 200:
        url = f"https://raw.githubusercontent.com/{repo}/master/{filepath}"
        r = requests.get(url, headers=headers, stream=True)

    if r.status_code == 200:
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return dest_path

    raise RuntimeError(f"Download failed: {r.status_code}")
# ---------- Load model (with caching) ----------
@st.cache_resource
def get_model(local_name, repo_path):
    local_file = CACHE_DIR / local_name
    if not local_file.exists():
        download_from_github(PRIVATE_REPO, repo_path, local_file)
    return YOLO(str(local_file))

# ---------- Inference helper ----------
def run_yolo_predict(model, img):
    # LOWER CONFIDENCE (0.1) helps catch faint cracks
    results = model.predict(source=img, imgsz=1024, conf=0.1, verbose=False)
    return results[0]

# ---------- Measurement Logic ----------
def mask_max_width_pixels(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_width = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        max_width = max(max_width, w)
    return max_width

# ---------- Streamlit UI ----------
st.title("🏗️ Crack Width Measurement")
st.markdown("Upload an image with a **crack** and a **reference coin**.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    st.image(image, caption="Original Image", use_container_width=True)

    try:
        with st.status("Loading private models...") as status:
            crack_model = get_model("crack_model.pt", CRACK_MODEL_PATH)
            coin_model = get_model("coin_model.pt", COIN_MODEL_PATH)
            status.update(label="Models Ready!", state="complete")
            
        # --- PREDICTIONS ---
        res_coin = run_yolo_predict(coin_model, img_np)
        res_crack = run_yolo_predict(crack_model, img_np)

        # --- DEBUG VIEW ---
        st.subheader("🔍 Model Detections (Debug)")
        col1, col2 = st.columns(2)
        col1.image(res_coin.plot()[:,:,::-1], caption="Coin Detection", use_container_width=True)
        col2.image(res_crack.plot()[:,:,::-1], caption="Crack Detection", use_container_width=True)

        # --- CALCULATIONS ---
        coin_px = None
        if len(res_coin.boxes) > 0:
            x1, y1, x2, y2 = res_coin.boxes.xyxy[0].cpu().numpy().astype(int)
            coin_px = x2 - x1

        crack_px = None
        if res_crack.masks is not None and len(res_crack.masks.data) > 0:
                mask_data = res_crack.masks.data.cpu().numpy()[0]
                mask_uint8 = (mask_data * 255).astype("uint8")
                crack_px = mask_max_width_pixels(mask_uint8)
        
        if coin_px and crack_px:
            mm_per_pixel = COIN_DIAMETER_MM / coin_px
            width_mm = crack_px * mm_per_pixel
            
            st.metric("Estimated Crack Width", f"{width_mm:.2f} mm")
            
            # Severity Rating
            if width_mm < 0.1: severity = "Hairline (Safe)"
            elif width_mm < 0.3: severity = "Minor"
            else: severity = "Severe (Action Required)"
            st.info(f"Condition: **{severity}**")
        else:
            st.error("⚠️ Detection Failed. Look at the 'Debug' images above to see which object was missed.")

    except Exception as e:
        st.error(f"Error: {e}")
