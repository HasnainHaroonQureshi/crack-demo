import os
import io
import time
import requests
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch

# Optional: ultralytics YOLO class (if your model is compatible)
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

# ---------- CONFIG (set via Streamlit secrets) ----------
GITHUB_USER = st.secrets.get("github_user")
GITHUB_TOKEN = st.secrets.get("github_token")
PRIVATE_REPO = st.secrets.get("private_repo")  # e.g., "youruser/models-private"
CRACK_MODEL_PATH = st.secrets.get("crack_model_path", "crack_yolov11s_seg.pt")
COIN_MODEL_PATH  = st.secrets.get("coin_model_path", "coin_yolov11s_seg.pt")
COIN_DIAMETER_MM = float(st.secrets.get("coin_diameter_mm", "20.0"))  # set actual coin diameter

CACHE_DIR = Path("models_cache")
CACHE_DIR.mkdir(exist_ok=True)

# ---------- Helper: download file from private GitHub repo ----------
def download_from_github(repo, filepath, dest_path):
    """
    Download a file from a private GitHub repo using a PAT.
    """
    url = f"https://api.github.com/repos/{repo}/contents/{filepath}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        raise RuntimeError(f"GitHub download failed: {r.status_code} {r.text}")
    data = r.json()
    import base64
    content = base64.b64decode(data["content"])
    with open(dest_path, "wb") as f:
        f.write(content)
    return dest_path

# ---------- Load model (with caching) ----------
def get_model(local_name, repo_path):
    local_file = CACHE_DIR / local_name
    if not local_file.exists():
        with st.spinner(f"Downloading {local_name} from private repo..."):
            download_from_github(PRIVATE_REPO, repo_path, local_file)
    # Try ultralytics first
    if ULTRALYTICS_AVAILABLE:
        try:
            model = YOLO(str(local_file))
            return ("ultralytics", model)
        except Exception:
            pass
    # Fallback to torch.load
    model = torch.load(str(local_file), map_location="cpu")
    model.eval()
    return ("torch", model)

# ---------- Inference helpers ----------
def run_ultralytics_detect(model, img):
    # model.predict returns results; adapt depending on ultralytics version
    results = model.predict(source=img, imgsz=1024, conf=0.25, verbose=False)
    # results is a list; take first
    r = results[0]
    # bounding boxes: r.boxes.xyxy, r.boxes.conf, r.boxes.cls
    # masks: r.masks.data (if segmentation)
    return r

def run_torch_forward(model, img_tensor):
    # This is a fallback; exact forward depends on how model was saved.
    with torch.no_grad():
        out = model(img_tensor)
    return out

# ---------- Utility: compute pixel width from segmentation mask ----------
def mask_max_width_pixels(mask):
    """
    mask: binary mask (H,W) uint8
    returns: maximum width in pixels across mask (approximate)
    """
    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_width = 0
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        max_width = max(max_width, w)
    return max_width

# ---------- Pixel to mm conversion ----------
def pixels_to_mm(crack_pixels, coin_pixels, coin_diameter_mm):
    if coin_pixels <= 0:
        return None
    return (crack_pixels / coin_pixels) * coin_diameter_mm

# ---------- Streamlit UI ----------
st.title("Crack Width Measurement (demo)")

st.markdown("Upload an image containing a crack and a reference coin. The app detects the coin and the crack, then computes crack width in mm.")

uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    st.image(image, caption="Input image", use_column_width=True)

    # Load models (cached)
    try:
        status = st.empty()
        status.text("Loading models (may take a few seconds)...")
        crack_loader, crack_model = get_model("crack_model.pt", CRACK_MODEL_PATH)
        coin_loader, coin_model   = get_model("coin_model.pt", COIN_MODEL_PATH)
        status.text("Models loaded.")
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

    # Run coin detection
    status.text("Running coin detection...")
    coin_pixels = None
    coin_box = None
    if coin_loader == "ultralytics":
        r = run_ultralytics_detect(coin_model, img_np)
        # find coin class index or assume single detection
        if hasattr(r, "boxes") and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            # take the largest box (by area)
            areas = (xyxy[:,2]-xyxy[:,0])*(xyxy[:,3]-xyxy[:,1])
            idx = int(np.argmax(areas))
            x1,y1,x2,y2 = xyxy[idx].astype(int)
            coin_box = (x1,y1,x2,y2)
            coin_pixels = x2 - x1
    else:
        # Fallback: user must implement custom forward and parsing
        st.warning("Coin model is not ultralytics-compatible; fallback not implemented in demo.")
    # Run crack segmentation
    status.text("Running crack segmentation...")
    crack_pixels = None
    if crack_loader == "ultralytics":
        r = run_ultralytics_detect(crack_model, img_np)
        # get mask (if available)
        if hasattr(r, "masks") and r.masks is not None:
            # r.masks.data is (N, H, W) or similar
            mask_data = r.masks.data.cpu().numpy()
            # choose largest mask
            areas = mask_data.reshape(mask_data.shape[0], -1).sum(axis=1)
            idx = int(np.argmax(areas))
            mask = (mask_data[idx] * 255).astype("uint8")
            # compute max width in pixels
            crack_pixels = mask_max_width_pixels(mask)
        else:
            st.warning("No segmentation masks returned by crack model.")
    else:
        st.warning("Crack model is not ultralytics-compatible; fallback not implemented in demo.")

    # Compute mm
    if coin_pixels and crack_pixels:
        width_mm = pixels_to_mm(crack_pixels, coin_pixels, COIN_DIAMETER_MM)
        st.success(f"Estimated crack width: **{width_mm:.2f} mm**")
        # severity example
        if width_mm < 0.10:
            severity = "Hairline"
        elif width_mm < 0.30:
            severity = "Minor"
        elif width_mm < 0.50:
            severity = "Moderate"
        else:
            severity = "Severe"
        st.write(f"Severity: **{severity}**")
        # Annotate and show image
        vis = img_np.copy()
        if coin_box:
            x1,y1,x2,y2 = coin_box
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(vis, f"coin_px={coin_pixels}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        if 'mask' in locals():
            color_mask = np.zeros_like(vis)
            color_mask[mask>0] = (0,0,255)
            vis = cv2.addWeighted(vis, 0.8, color_mask, 0.4, 0)
        st.image(vis[:,:,::-1], caption="Annotated result", use_column_width=True)
    else:
        st.error("Could not compute width. Ensure coin and crack are visible and models are compatible.")
