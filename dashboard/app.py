import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import streamlit as st
import cv2
import numpy as np
import torch
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
    model = UNet().to(device)
    model.load_state_dict(torch.load("E:\\Underwater Image Enhancment\\model\\unet_best.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# ---------------- UI ---------------- #
st.title("🌊 Underwater Image Enhancement System")
st.markdown(
    """
    This application enhances underwater images using a **U-Net based deep learning model**\
    trained on the **EUVP dataset**.
    """
)

uploaded_file = st.file_uploader(
    "Upload an underwater image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display original
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image_rgb, use_column_width=True)

    # Preprocess
    img_resized = cv2.resize(image_rgb, (256, 256)) / 255.0
    tensor = torch.tensor(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # Inference
    with st.spinner("Enhancing image..."):
        with torch.no_grad():
            output = model(tensor)

    # Postprocess
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = (output * 255).astype(np.uint8)

    with col2:
        st.subheader("Enhanced Image")
        st.image(output, use_column_width=True)

    # Download option
    result_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', result_bgr)

    st.download_button(
        label="⬇️ Download Enhanced Image",
        data=buffer.tobytes(),
        file_name="enhanced_image.png",
        mime="image/png"
    )
    uiqm_score = compute_uiqm(output)
    st.success(f"UIQM Score: {uiqm_score:.4f}")

# ---------------- Footer ---------------- #
st.markdown("---")
st.markdown(
    "**Model:** U-Net  |  **Dataset:** EUVP  |  **Metrics:** PSNR, SSIM, UIQM"
)
# ================= ABOUT PROJECT =================
st.markdown("---")
st.header("📘 About This Project")

st.markdown("""
### 🌊 AI-Based Underwater Image Enhancement System

This project enhances underwater images using a **deep learning U-Net model** trained on the **EUVP dataset**.  
Underwater images often suffer from:

- Color distortion
- Low contrast
- Poor visibility
- Light absorption effects

Our model learns to restore natural colors and improve clarity automatically.

---

### 🧠 Model Used
- **Architecture:** U-Net (Encoder–Decoder CNN)
- **Loss Function:** L1 Loss
- **Framework:** PyTorch
- **Training Platform:** Google Colab (GPU)

---

### 📊 Dataset
- **Dataset Name:** EUVP (Enhancing Underwater Visual Perception)
- **Type:** Paired underwater images
- **Categories Used:**
  - Underwater Dark
  - Underwater Imagenet
  - Underwater Scenes

---

### 📈 Evaluation Metrics
- **PSNR** → Measures reconstruction quality
- **SSIM** → Measures structural similarity
- **UIQM** → Underwater image quality measure

Higher values indicate better enhancement.

---

### 🌍 Real World Applications
- Marine research & ocean exploration
- Underwater robotics
- Archaeological surveys
- Surveillance systems
- Photography enhancement

---

### 🚀 Future Scope
- Real-time video enhancement
- Mobile application deployment
- Transformer-based enhancement models
- GAN-based underwater restoration
- Edge device optimization

---

### 👨‍💻 Developed By
Atul Kushwah — AI Image Enhancement
""")
