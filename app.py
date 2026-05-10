import os
import requests
import numpy as np
import cv2
import torch
import streamlit as st
from PIL import Image
from pathlib import Path
from ultralytics import YOLO

# =========================================================
# CONFIG
# =========================================================

GITHUB_TOKEN = st.secrets.get("github_token")
PRIVATE_REPO = st.secrets.get("private_repo")

CRACK_MODEL_PATH = st.secrets.get(
    "crack_model_path",
    "Crackdetection_model.pt"
)

COIN_MODEL_PATH = st.secrets.get(
    "coin_model_path",
    "Coindetection_model.pt"
)

COIN_DIAMETER_MM = float(
    st.secrets.get("coin_diameter_mm", "18.5")
)

CACHE_DIR = Path("models_cache")
CACHE_DIR.mkdir(exist_ok=True)

# =========================================================
# DOWNLOAD MODEL FROM PRIVATE GITHUB
# =========================================================

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

    raise RuntimeError(
        f"Download failed: {r.status_code}"
    )

# =========================================================
# LOAD YOLO MODELS
# =========================================================

@st.cache_resource
def get_model(local_name, repo_path):

    local_file = CACHE_DIR / local_name

    if not local_file.exists():
        download_from_github(
            PRIVATE_REPO,
            repo_path,
            local_file
        )

    return YOLO(str(local_file))

# =========================================================
# YOLO PREDICTION
# =========================================================

def run_yolo_predict(model, img):

    results = model.predict(
        source=img,
        imgsz=1024,
        conf=0.25,
        retina_masks=True,
        verbose=False
    )

    return results[0]

# =========================================================
# IMPROVED WIDTH CALCULATION
# =========================================================

def mask_max_width_pixels(mask):

    mask = (mask > 0).astype(np.uint8)

    dist = cv2.distanceTransform(
        mask,
        cv2.DIST_L2,
        5
    )

    max_width = dist.max() * 2

    return max_width

# =========================================================
# COMBINE ALL CRACK MASKS
# =========================================================

def create_clean_mask(res_crack):

    if res_crack.masks is None:
        return None

    masks = res_crack.masks.data.cpu().numpy()

    if len(masks) == 0:
        return None

    combined_mask = np.zeros(
        (
            masks[0].shape[0],
            masks[0].shape[1]
        ),
        dtype=np.uint8
    )

    for mask in masks:

        binary = (mask > 0.5).astype(np.uint8) * 255

        combined_mask = cv2.bitwise_or(
            combined_mask,
            binary
        )

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)

    combined_mask = cv2.morphologyEx(
        combined_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    combined_mask = cv2.medianBlur(
        combined_mask,
        5
    )

    return combined_mask

# =========================================================
# DRAW COMBINED RESULTS
# =========================================================

def draw_combined_results(
    img,
    res_coin,
    crack_mask,
    width_mm=None
):

    output = img.copy()

    # ---------------- COIN BOX ----------------

    if len(res_coin.boxes) > 0:

        for box in res_coin.boxes.xyxy.cpu().numpy():

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            cv2.putText(
                output,
                "Coin",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    # ---------------- CRACK MASK ----------------

    if crack_mask is not None:

        red_mask = np.zeros_like(output)

        red_mask[:, :, 2] = crack_mask

        output = cv2.addWeighted(
            output,
            1.0,
            red_mask,
            0.4,
            0
        )

    # ---------------- WIDTH TEXT ----------------

    if width_mm is not None:

        cv2.putText(
            output,
            f"Width: {width_mm:.2f} mm",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 0),
            3
        )

    return output

# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(
    page_title="Crack Width Measurement",
    layout="wide"
)

st.title("🏗️ Crack Width Measurement")

st.markdown(
    """
Upload an image containing:

- A structural crack
- A reference coin
"""
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# =========================================================
# PROCESS IMAGE
# =========================================================

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    img_np = np.array(image)

    st.image(
        img_np,
        caption="Original Image",
        use_container_width=True
    )

    try:

        # ---------------- LOAD MODELS ----------------

        with st.status(
            "Loading AI Models..."
        ) as status:

            crack_model = get_model(
                "crack_model.pt",
                CRACK_MODEL_PATH
            )

            coin_model = get_model(
                "coin_model.pt",
                COIN_MODEL_PATH
            )

            status.update(
                label="Models Loaded Successfully!",
                state="complete"
            )

        # ---------------- PREDICTIONS ----------------

        res_coin = run_yolo_predict(
            coin_model,
            img_np
        )

        res_crack = run_yolo_predict(
            crack_model,
            img_np
        )

        # =================================================
        # COIN DETECTION
        # =================================================

        coin_px = None

        if len(res_coin.boxes) > 0:

            # Largest coin
            boxes = res_coin.boxes.xyxy.cpu().numpy()

            largest_box = max(
                boxes,
                key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
            )

            x1, y1, x2, y2 = map(
                int,
                largest_box
            )

            # Use average diameter
            coin_px = (
                ((x2 - x1) + (y2 - y1)) / 2
            )

        # =================================================
        # CRACK MASK
        # =================================================

        combined_mask = create_clean_mask(
            res_crack
        )

        crack_px = None

        if combined_mask is not None:

            crack_px = mask_max_width_pixels(
                combined_mask
            )

        # =================================================
        # WIDTH CALCULATION
        # =================================================

        width_mm = None

        if coin_px and crack_px:

            mm_per_pixel = (
                COIN_DIAMETER_MM / coin_px
            )

            width_mm = (
                crack_px * mm_per_pixel
            )

        # =================================================
        # DRAW FINAL RESULT
        # =================================================

        result_image = draw_combined_results(
            img_np,
            res_coin,
            combined_mask,
            width_mm
        )

        st.subheader("Detection Result")

        st.image(
            result_image,
            caption="Combined AI Detection",
            use_container_width=True
        )

        # =================================================
        # RESULTS
        # =================================================

        if width_mm is not None:

            st.metric(
                "Estimated Crack Width",
                f"{width_mm:.2f} mm"
            )

            # Severity classification
            if width_mm < 0.1:
                severity = "Hairline (Safe)"

            elif width_mm < 0.3:
                severity = "Minor"

            else:
                severity = "Severe (Action Required)"

            st.info(
                f"Condition: **{severity}**"
            )

        else:

            st.error(
                """
Detection failed.

Possible reasons:
- Coin not detected
- Crack not segmented properly
- Poor image quality
"""
            )

    except Exception as e:

        st.error(
            f"Error: {e}"
        )
