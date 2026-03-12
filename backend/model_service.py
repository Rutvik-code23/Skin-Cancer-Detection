import torch
import torch.nn as nn
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
import os
from PIL import Image
import numpy as np

MODEL_PATH = "model/skin_cancer_vit.pth"
PRETRAINED_NAME = "google/vit-base-patch16-224"
CLASS_NAMES = ["Cancerous", "Non-Cancerous"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Processor and Transform
processor = ViTImageProcessor.from_pretrained(PRETRAINED_NAME)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

# Lazy load model
_model = None

def get_model():
    global _model
    if _model is None:
        _model = ViTForImageClassification.from_pretrained(PRETRAINED_NAME)
        _model.classifier = nn.Linear(_model.classifier.in_features, len(CLASS_NAMES))
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=device)
            _model.load_state_dict(state_dict)
        _model.to(device)
        _model.eval()
    return _model

def predict_image(image_path: str):
    model = get_model()
    pil_img = Image.open(image_path).convert("RGB")
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        
    cancer_prob = float(probs[0]) # Assuming 0 is Cancerous, 1 is Non-Cancerous, adjust if needed
    
    # "Cancerous" -> high risk if confidence > 0.6
    # "Non-Cancerous" -> low risk
    risk_level = "Medium"
    if CLASS_NAMES[pred_idx] == "Cancerous":
        if cancer_prob > 0.8:
            risk_level = "High"
        else:
            risk_level = "Medium"
    else:
        risk_level = "Low"
        
    return {
        "prediction": "Melanoma" if CLASS_NAMES[pred_idx] == "Cancerous" else "Benign", # Mapped to UI names
        "confidence": round(float(probs[pred_idx]) * 100, 2),
        "riskLevel": risk_level
    }
