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
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Concrete Crack Detection AI",
    page_icon="🏗️",
    layout="wide"
)

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

DEFAULT_COIN_DIAMETER_MM = float(
    st.secrets.get("coin_diameter_mm", "18.5")
)

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
    step=0.1
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=1.0,
    value=0.25,
    step=0.05
)

iou_threshold = st.sidebar.slider(
    "IoU Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.45,
    step=0.05
)

st.sidebar.title("🧪 Width Calibration")

calibration_factor = st.sidebar.slider(
    "Calibration Factor",
    min_value=0.10,
    max_value=3.00,
    value=1.00,
    step=0.01
)

st.sidebar.caption(
    "Final Width = Measured Width × Calibration Factor"
)

# =========================================================
# CALIBRATION SETTINGS
# =========================================================

st.sidebar.title("📏 Calibration")

use_manual_calibration = st.sidebar.checkbox(
    "Use Manual Calibration",
    value=False
)

manual_mm_per_pixel = st.sidebar.number_input(
    "Manual mm/pixel Calibration",
    min_value=0.0001,
    max_value=10.0,
    value=0.05,
    step=0.001,
    format="%.4f"
)

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

def run_yolo_predict(
    model,
    img,
    conf_threshold,
    iou_thresh
):

    results = model.predict(
        source=img,
        imgsz=1024,
        conf=conf_threshold,
        iou=iou_thresh,
        retina_masks=True,
        verbose=False
    )

    return results[0]

# =========================================================
# WIDTH CALCULATION
# =========================================================

# =========================================================
# WIDTH + MAX WIDTH LOCATION
# =========================================================

def mask_max_width_pixels(mask):

    mask = (mask > 0).astype(np.uint8)

    dist = cv2.distanceTransform(
        mask,
        cv2.DIST_L2,
        5
    )

    # Maximum radius
    max_radius = dist.max()

    # Crack width
    max_width = max_radius * 2

    # Location of maximum width
    max_loc = np.unravel_index(
        np.argmax(dist),
        dist.shape
    )

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

    combined_mask = np.zeros(
        (
            masks[0].shape[0],
            masks[0].shape[1]
        ),
        dtype=np.uint8
    )

    for mask in masks:

        binary = (
            mask > 0.5
        ).astype(np.uint8) * 255

        combined_mask = cv2.bitwise_or(
            combined_mask,
            binary
        )

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
# DRAW RESULTS
# =========================================================

# =========================================================
# DRAW RESULTS
# =========================================================

def draw_combined_results(
    img,
    res_coin,
    crack_mask,
    width_mm=None,
    max_width_point=None
):

    output = img.copy()

    # =====================================================
    # COIN BOX
    # =====================================================

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

    # =====================================================
    # CRACK MASK
    # =====================================================

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

    # =====================================================
    # MAX WIDTH POINT
    # =====================================================

    if max_width_point is not None:

        x, y = max_width_point

        # Green dot
        cv2.circle(
            output,
            (x, y),
            10,
            (0, 255, 0),
            -1
        )

        # Label
        cv2.putText(
            output,
            "Max Width",
            (x + 15, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # =====================================================
    # WIDTH TEXT
    # =====================================================

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
# MAIN TITLE
# =========================================================

st.title("🏗️ Concrete Crack Detection AI")

st.markdown("""
AI-powered structural crack detection and severity assessment using YOLOv11 segmentation models.
""")

# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Concrete Image",
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
        caption="Uploaded Image",
        use_container_width=True
    )

    try:

        # Load Models
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

        # Predictions
        res_coin = run_yolo_predict(
            coin_model,
            img_np,
            confidence_threshold,
            iou_threshold
        )

        res_crack = run_yolo_predict(
            crack_model,
            img_np,
            confidence_threshold,
            iou_threshold
        )

        # =================================================
        # COIN DETECTION
        # =================================================

        coin_px = None

        if len(res_coin.boxes) > 0:

            boxes = res_coin.boxes.xyxy.cpu().numpy()

            largest_box = max(
                boxes,
                key=lambda b:
                (b[2] - b[0]) * (b[3] - b[1])
            )

            x1, y1, x2, y2 = map(
                int,
                largest_box
            )

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
        max_width_point = None

        if combined_mask is not None:

            crack_px, max_width_point = mask_max_width_pixels(
            combined_mask
        )

        # =================================================
        # CALIBRATION
        # =================================================

        width_mm = None
        mm_per_pixel = None

        if use_manual_calibration:

            mm_per_pixel = manual_mm_per_pixel

        elif coin_px:

            mm_per_pixel = (
                coin_diameter_mm / coin_px
            )

        if mm_per_pixel is not None and crack_px:

        # Raw measured width
        measured_width = crack_px * mm_per_pixel

        # Calibrated width
        width_mm = measured_width * calibration_factor

        # =================================================
        # DRAW RESULTS
        # =================================================

        result_image = draw_combined_results(
            img_np,
            res_coin,
            combined_mask,
            width_mm,
            max_width_point
        )

        st.subheader("Detection Result")

        st.image(
            result_image,
            caption="AI Detection Output",
            use_container_width=True
        )

        # =================================================
        # METRICS
        # =================================================

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Confidence",
                f"{confidence_threshold:.2f}"
            )

        with col2:

            st.metric(
                "IoU Threshold",
                f"{iou_threshold:.2f}"
            )

        with col3:

            if mm_per_pixel is not None:

                st.metric(
                    "Calibration",
                    f"{mm_per_pixel:.4f} mm/pixel"
                )


# =================================================
# WIDTH CALIBRATION INFO
# =================================================

col4, col5 = st.columns(2)

with col4:

    st.metric(
        "Calibration Factor",
        f"{calibration_factor:.2f}"
    )

with col5:

    if width_mm is not None:

        st.metric(
            "Calibrated Width",
            f"{width_mm:.2f} mm"
        )
        # =================================================
        # WIDTH + SEVERITY
        # =================================================

        if width_mm is not None:

            st.metric(
                "Estimated Crack Width",
                f"{width_mm:.2f} mm"
            )

            # Severity Classification

            if width_mm < 0.1:

                severity = "Hairline"

                description = (
                    "Barely visible; usually non-structural; "
                    "monitor periodically."
                )

                recommendation = (
                    "Routine monitoring recommended."
                )

            elif 0.1 <= width_mm < 0.3:

                severity = "Minor / Fine"

                description = (
                    "Generally acceptable in dry/interior areas; "
                    "surface sealing may be required in wet environments."
                )

                recommendation = (
                    "Consider protective sealing if exposed to moisture."
                )

            elif 0.3 <= width_mm <= 0.5:

                severity = "Moderate"

                description = (
                    "Exceeds most ACI crack-width limits; "
                    "may indicate overstress or durability concerns."
                )

                recommendation = (
                    "Engineering evaluation recommended."
                )

            else:

                severity = "Severe"

                description = (
                    "Significant distress with possible risk of "
                    "corrosion or structural deterioration."
                )

                recommendation = (
                    "Immediate repair required "
                    "(e.g., epoxy injection or rehabilitation)."
                )

            st.subheader("Structural Condition Assessment")

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "Severity Level",
                    severity
                )

            with c2:

                st.metric(
                    "Measured Width",
                    f"{width_mm:.2f} mm"
                )

            st.warning(
                f"**Description:** {description}"
            )

            st.success(
                f"**Recommended Action:** {recommendation}"
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

        st.error(f"Error: {e}")

# =========================================================
# ABOUT SECTION
# =========================================================

st.markdown("---")

st.header("📘 About the Project")

st.markdown("""
### Project Overview

This project focuses on the development of an advanced deep learning system for automated crack detection and severity classification using **YOLOv11**. Traditional structural assessments require manual visual inspections, which can be time-consuming, subjective, and difficult to scale across large infrastructures.

By leveraging state-of-the-art computer vision algorithms, **Concrete Crack Detection AI** achieves real-time detection capabilities with high precision. The system not only identifies cracks in concrete but also provides actionable insights through severity classification.

This allows engineers and maintenance teams to:

- Quickly assess structural integrity
- Prioritize repair work
- Reduce maintenance costs
- Improve inspection efficiency
- Enhance public safety

This research highlights the critical importance of integrating AI technologies into civil engineering, showcasing how emerging tools can transform and modernize traditional infrastructure monitoring methods.
""")

st.markdown("---")

st.header("👨‍🔬 Research Team")

st.markdown("""
### Project Lead
**Hasnain Haroon**

Developed as a Final Year Project (FYP) within the Department of Civil Engineering at COMSATS University Islamabad, Wah Campus.

The project emphasizes practical implementation and collaboration in modern engineering education.

---

### Project Supervisor
**Engr. Sandeerah Choudhary**

---

### Key Team Members
- **Liza Liaqat**
- **Ammar Faheem Khawaja**
""")
