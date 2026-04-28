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


# ---------------- UI HEADER ---------------- #
st.markdown("""
<style>
.main-title {
    font-size: 32px;
    font-weight: bold;
    color: #00bcd4;
}
.subtitle {
    font-size: 16px;
    color: #aaa;
}
.metric-box {
    padding: 10px;
    border-radius: 10px;
    background-color: #111;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🌊 Underwater Image Enhancement System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enhance underwater images using deep learning (U-Net)</p>', unsafe_allow_html=True)


# ---------------- SIDEBAR ---------------- #
st.sidebar.header("⚙️ Controls")

show_graph = st.sidebar.checkbox("Show Graph", True)
show_metrics = st.sidebar.checkbox("Show Metrics", True)
image_size = st.sidebar.slider("Resize Image", 128, 512, 256)

st.sidebar.markdown("---")
st.sidebar.info("Model: U-Net\nDataset: EUVP\nMetric: UIQM")


# ---------------- FILE UPLOAD ---------------- #
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize BOTH images (important fix)
    image_resized = cv2.resize(image_rgb, (image_size, image_size)) / 255.0

    tensor = torch.tensor(image_resized)\
        .permute(2, 0, 1)\
        .unsqueeze(0)\
        .float()\
        .to(device)

    # ---------------- INFERENCE ---------------- #
    with st.spinner("🚀 Enhancing image..."):
        with torch.no_grad():
            output = model(tensor)

    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)

    original_display = cv2.resize(image_rgb, (image_size, image_size))

    # ---------------- TABS ---------------- #
    tab1, tab2, tab3 = st.tabs(["🔍 Comparison", "📊 Metrics", "📄 Details"])

    # ================= TAB 1 ================= #
    with tab1:
        st.subheader("Before vs After")

        image_comparison(
            img1=Image.fromarray(original_display),
            img2=Image.fromarray(output),
            label1="Original",
            label2="Enhanced",
            width=700
        )

        # Side-by-side view (extra premium)
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_display, caption="Original", use_column_width=True)
        with col2:
            st.image(output, caption="Enhanced", use_column_width=True)

        # Download
        result_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".png", result_bgr)

        st.download_button(
            "⬇️ Download Enhanced Image",
            buffer.tobytes(),
            file_name="enhanced.png"
        )


    # ================= TAB 2 ================= #
    with tab2:

        if show_metrics:
            original_uiqm = compute_uiqm(original_display)
            enhanced_uiqm = compute_uiqm(output)
            improvement = enhanced_uiqm - original_uiqm

            st.subheader("📊 Image Quality")

            col1, col2, col3 = st.columns(3)

            col1.metric("Original", f"{original_uiqm:.2f}")
            col2.metric("Enhanced", f"{enhanced_uiqm:.2f}")
            col3.metric("Improvement", f"{improvement:.2f}")

            # Status message
            if improvement > 0:
                st.success("✅ Image quality improved")
            else:
                st.warning("⚠️ No significant improvement")


        # Graph
        if show_graph:
            st.subheader("📈 Quality Comparison")

            fig, ax = plt.subplots(figsize=(6,4))

            labels = ["Original", "Enhanced"]
            values = [original_uiqm, enhanced_uiqm]

            bars = ax.bar(labels, values, color=["#ff6b6b", "#00b894"])

            ax.set_title("UIQM Comparison")
            ax.grid(axis="y", linestyle="--", alpha=0.3)

            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height(),
                        f"{bar.get_height():.2f}",
                        ha="center")

            st.pyplot(fig)


    # ================= TAB 3 ================= #
    with tab3:

        st.subheader("🧠 Enhancement Details")

        st.markdown(f"""
        - **Model Used:** U-Net  
        - **Input Resolution:** {image_rgb.shape[1]} × {image_rgb.shape[0]}  
        - **Processing Device:** {device}  
        """)

        st.markdown("### 📌 Interpretation")

        if improvement > 0:
            st.info("Enhanced image has better contrast and visibility.")
        else:
            st.info("Model struggled due to complex lighting conditions.")


# ---------------- FOOTER ---------------- #
st.markdown("---")
st.markdown("Built with using PyTorch & Streamlit")