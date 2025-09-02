# 🌽 DinoCorn: Corn Disease Classifier

DinoCorn is a deep learning application that uses **DINOv2** as a backbone with a custom classification head to classify corn leaf diseases into 4 categories:

- Healthy
- Gray Leaf Spot
- Common Rust
- Northern Leaf Blight

The project includes a **Streamlit** app for interactive deployment.

---

## 👨‍💻 Developers
- Yasir Ahmad  
- Naveen Nidadavolu  
- Kumar Shivam  
- Vinothini A  

---

## 🚀 Features
- Transfer Learning using **DINOv2 (Facebook AI)**
- Fine-tuned model for **Corn Disease Classification**
- Streamlit-based interactive web app
- Shows predictions with confidence scores

---

## 📂 Project Structure
```

.
├── main.py                # Streamlit app
├── dinov2\_finetuned.pth   # Fine-tuned model weights
├── requirements.txt       # Dependencies
└── README.md              # Project documentation

```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/dinocorn.git
cd dinocorn
````

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the model weights

Ensure the fine-tuned weights file `dinov2_finetuned.pth` is in the project root directory.

If you don’t have the file, you need to train DinoCorn first or request the weights from the authors.

### 5. Run the app

```bash
streamlit run main.py
```

### 6. Open in browser

Streamlit will start a local server. Open the link shown in the terminal (usually `http://localhost:8501`).

---

## 🖼️ Usage

1. Upload a corn leaf image (`.jpg`, `.png`, `.jpeg`).
2. The app will classify it into one of the four classes.
3. The prediction and confidence score will be displayed.

---

## 🛠️ Tech Stack

* **Python 3.9+**
* **PyTorch**
* **Hugging Face Transformers**
* **Streamlit**
* **Torchvision**

---

## 📜 License

This project is developed for research and educational purposes. Please contact the authors for further usage permissions.

