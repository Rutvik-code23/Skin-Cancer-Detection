import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
import matplotlib.pyplot as plt

# ================================
# Config
# ================================
MODEL_PATH = "skin_cancer_vit.pth"
PRETRAINED_NAME = "google/vit-base-patch16-224"
CLASS_NAMES = ["Cancerous", "Non-Cancerous"]

# ================================
# Streamlit App
# ================================
st.set_page_config(page_title="Skin Cancer Detection — ViT", page_icon="🩺", layout="wide")
st.title("🩺 Skin Cancer Detection — ViT")
st.markdown("Upload a lesion/mole photo.")

# Load model + processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViTForImageClassification.from_pretrained(PRETRAINED_NAME)
model.classifier = nn.Linear(model.classifier.in_features, len(CLASS_NAMES))
if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
model.to(device)
model.eval()

processor = ViTImageProcessor.from_pretrained(PRETRAINED_NAME)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

# Sidebar
st.sidebar.header("Settings & Options")
show_heatmap = st.sidebar.checkbox("Show saliency heatmap (explainability)", value=True)
use_camera = st.sidebar.checkbox("Use camera input (if available)", value=False)

st.sidebar.markdown("---")
st.sidebar.write(f"*Model:* {PRETRAINED_NAME}")
st.sidebar.write(f"*Device:* {device}")

col1, col2 = st.columns([1, 1])

with col1:
    if use_camera:
        cam_file = st.camera_input("Take a photo")
    else:
        cam_file = None

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

with col2:
    st.markdown("Quick Notes & Disclaimer")
    st.write(
        "- This is a research/demo tool and *not* a medical diagnosis.\n"
        "- Results may be incorrect for different image quality, lighting, or lesion types."
    )
    st.info("Tip: Use clear, well-lit photos focusing on the lesion.")

# Collect image
image_files = []
if cam_file is not None:
    pil = Image.open(cam_file).convert("RGB")
    image_files.append(("camera_input", pil))

if uploaded_file:
    pil = Image.open(uploaded_file).convert("RGB")
    image_files.append((getattr(uploaded_file, "name", "uploaded"), pil))

if len(image_files) == 0:
    st.stop()

# Results
results = []

fname, pil_img = image_files[0]
display_img = ImageEnhance.Sharpness(pil_img).enhance(1.1)

st.markdown("---")
st.subheader(f"File: {fname}")

colA, colB = st.columns([1, 1])
with colA:
    st.image(display_img, caption="Input Image", use_container_width=True)

# Prediction
img_tensor = transform(pil_img).unsqueeze(0).to(device)
with torch.no_grad():
    outputs = model(img_tensor)
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))

cancer_prob = float(probs[1])

with colB:
    st.metric("Predicted label", CLASS_NAMES[pred_idx])
    # st.write(f"*Cancerous probability:* {cancer_prob:.3f}")
    fig, ax = plt.subplots(figsize=(4, 2.2))
    vals = np.array(probs)
    idx = np.arange(len(CLASS_NAMES))
    ax.barh(idx, vals)
    ax.set_yticks(idx)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Probability")
    ax.set_xlim(0, 1)
    st.pyplot(fig)

# Simple saliency
if show_heatmap:
    img_tensor.requires_grad_()
    outputs = model(img_tensor)
    logit = outputs.logits[0, pred_idx]
    logit.backward()

    grad = img_tensor.grad.detach().cpu().numpy()[0]  # C x H x W
    saliency = np.mean(np.abs(grad), axis=0)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-12)

    heatmap_resized = Image.fromarray(np.uint8(255 * saliency)).resize(pil_img.size).convert("L")
    heatmap_rgb = Image.merge("RGB", [heatmap_resized, Image.new("L", pil_img.size), Image.new("L", pil_img.size)])
    blended = Image.blend(pil_img.convert("RGB"), heatmap_rgb, alpha=0.45)

    st.image(blended, caption="Saliency heatmap overlay", use_container_width=True)

