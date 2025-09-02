import os
import torch
from torchvision import transforms
from PIL import Image
from transformers import Dinov2ForImageClassification
import streamlit as st

# ─────────────────────────────────────────────
# Fix: Use custom cache directory for HuggingFace
# ─────────────────────────────────────────────
CACHE_DIR = "/tmp/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

# ─────────────────────────────────────────────
# Configurations
# ─────────────────────────────────────────────
MODEL_PATH = "dinov2_finetuned.pth"  # your fine-tuned weights
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "healthy",
    "gray leaf spot",
    "common rust",
    "northern leaf blight",
]

# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Load base DinoV2 with classification head
    model = Dinov2ForImageClassification.from_pretrained(
        "facebook/dinov2-base",
        num_labels=len(CLASS_NAMES),
        cache_dir=CACHE_DIR
    )
    model.classifier = torch.nn.Linear(model.classifier.in_features, len(CLASS_NAMES))

    # Load fine-tuned weights
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=False)

    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ─────────────────────────────────────────────
# Image Preprocessing
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
def predict(image: Image.Image):
    img_t = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_t)
        logits = output.logits if hasattr(output, "logits") else output
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
    return CLASS_NAMES[pred_idx], confidence

# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="DinoCorn: Corn Disease Classifier", layout="centered")

st.title("🌽 DinoCorn")
st.subheader("Corn Disease Classification using DINOv2")

st.markdown("""
Upload an image of a corn leaf and DinoCorn will predict whether it is:

- **Healthy**
- **Gray Leaf Spot**
- **Common Rust**
- **Northern Leaf Blight**
""")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("🔍 Running inference...")
    label, conf = predict(image)
    st.success(f"Prediction: **{label}** ({conf*100:.2f}% confidence)")

# About section
st.sidebar.header("About DinoCorn")
st.sidebar.markdown("""
**DinoCorn** is a transfer-learning deep learning model that uses DINOv2 as the backbone 
and a custom classification head to classify corn leaf diseases into four categories.

**Developed by:**
- Yasir Ahmad
- Naveen Nidadavolu
- Kumar Shivam
- Vinothini A

**Built with:**
- DINOv2 (Facebook AI)
- PyTorch & Transformers
- Streamlit
""")

st.sidebar.caption("© 2025 DinoCorn Team")
