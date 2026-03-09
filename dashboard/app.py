import sys
import os

# Fix module import paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import streamlit as st
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_image_comparison import image_comparison

from model.unet import UNet
from utils.uiqm_single import compute_uiqm


# ---------------- Page Config ---------------- #
st.set_page_config(
    page_title="Underwater Image Enhancement",
    page_icon="🌊",
    layout="wide"
)


# ---------------- Load Model ---------------- #
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(ROOT_DIR, "model", "unet_best.pth")

    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, device


model, device = load_model()


# ---------------- UI ---------------- #
st.title("🌊 Underwater Image Enhancement System")

st.markdown("""
Enhance underwater images using a **U-Net deep learning model**
trained on the **EUVP dataset**.
""")


uploaded_file = st.file_uploader(
    "Upload an underwater image",
    type=["jpg", "jpeg", "png"]
)


# ---------------- Processing ---------------- #
if uploaded_file is not None:

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ---------------- Preprocess ---------------- #
    img_resized = cv2.resize(image_rgb, (256, 256)) / 255.0

    tensor = torch.tensor(img_resized)\
        .permute(2, 0, 1)\
        .unsqueeze(0)\
        .float()\
        .to(device)

    # ---------------- Inference ---------------- #
    with st.spinner("Enhancing image..."):
        with torch.no_grad():
            output = model(tensor)

    # ---------------- Postprocess ---------------- #
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)


    # ---------------- Before vs After Slider ---------------- #
    st.subheader("🔍 Before vs After Comparison")

    original_pil = Image.fromarray(image_rgb)
    enhanced_pil = Image.fromarray(output)

    image_comparison(
        img1=original_pil,
        img2=enhanced_pil,
        label1="Original",
        label2="Enhanced"
    )


    # ---------------- Download Button ---------------- #
    result_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", result_bgr)

    st.download_button(
        label="⬇️ Download Enhanced Image",
        data=buffer.tobytes(),
        file_name="enhanced_image.png",
        mime="image/png"
    )


    # ---------------- UIQM Metrics ---------------- #
    original_uiqm = compute_uiqm(image_rgb)
    enhanced_uiqm = compute_uiqm(output)
    improvement = enhanced_uiqm - original_uiqm

    st.subheader("📊 Image Quality Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric("Original UIQM", f"{original_uiqm:.2f}")
    col2.metric("Enhanced UIQM", f"{enhanced_uiqm:.2f}")
    col3.metric("Improvement", f"{improvement:.2f}")


    # ---------------- Graph ---------------- #
    st.subheader("📈 Quality Comparison")

    fig, ax = plt.subplots(figsize=(6,4))

    labels = ["Original", "Enhanced"]
    values = [original_uiqm, enhanced_uiqm]
    colors = ["#ff6b6b", "#00b894"]

    bars = ax.bar(labels, values, color=colors, width=0.5)

    ax.set_ylabel("UIQM Score")
    ax.set_title("Underwater Image Quality Improvement")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom"
        )

    st.pyplot(fig)


# ---------------- Footer ---------------- #
st.markdown("---")

st.markdown("""
**Model:** U-Net  
**Dataset:** EUVP  
**Metric:** UIQM  
""")