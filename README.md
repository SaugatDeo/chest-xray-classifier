# ChestAI — Medical Image Classification System

An AI-powered chest X-ray diagnostic assistant that classifies 14 pathological conditions using a custom deep learning model with explainable AI (Grad-CAM) visualizations. Fully deployed with a FastAPI backend and Streamlit frontend.

## 🌐 Live Demo

**Frontend:** https://chestai-diagnostic.streamlit.app  
**API Docs:** https://saugatiwi-chest-xray-classifier.hf.space/docs

Upload any chest X-ray and get:
- Probability scores for 14 disease classes
- Grad-CAM attention heatmap showing where the model focused
- Color-coded confidence bars (green/yellow/red)

---

## 🧠 Model Architecture

```
Input X-Ray (224×224×3)
        ↓
EfficientNet-B3 Backbone (pretrained ImageNet, partially fine-tuned)
        ↓
Custom Squeeze-and-Excitation (SE) Attention Block
        ↓
Global Average Pooling
        ↓
Dense(512) → BatchNorm → ReLU → Dropout(0.4)
        ↓
Dense(256) → BatchNorm → ReLU → Dropout(0.3)
        ↓
Dense(14, Sigmoid) — Multi-label output
        ↓
14 independent probability scores
```

**Why EfficientNet-B3?** Achieves better accuracy than ResNet50 with 4× fewer parameters — standard backbone for medical imaging in 2026.

**Why SE Attention?** Channel recalibration focuses the model on diagnostically relevant feature maps, improving sensitivity for rare conditions.

**Why Multi-label?** Real chest X-rays often show multiple co-occurring conditions. Binary classifiers miss this clinical reality.

---

## 📊 Training Results

| Metric | Value |
|---|---|
| Dataset | NIH ChestX-ray14 |
| Training Images | 86,524 |
| Test Images | 25,596 |
| Training Epochs | 5 |
| **Mean AUC-ROC** | **0.7933** |
| Best Class (Emphysema) | 0.8996 |
| Optimizer | Adam (lr=1e-4, weight_decay=1e-5) |
| Loss | BCEWithLogitsLoss + class weights |
| Scheduler | CosineAnnealingLR |

### Per-Class AUC-ROC Scores

| Disease | AUC-ROC |
|---|---|
| Emphysema | 0.8996 |
| Hernia | 0.8833 |
| Cardiomegaly | 0.8647 |
| Pneumothorax | 0.8606 |
| Edema | 0.8436 |
| Fibrosis | 0.8125 |
| Effusion | 0.8113 |
| Mass | 0.7853 |
| Pleural Thickening | 0.7472 |
| Atelectasis | 0.7423 |
| Consolidation | 0.7383 |
| Nodule | 0.7130 |
| Pneumonia | 0.7049 |
| Infiltration | 0.6997 |

> Comparable to published EfficientNet baselines (0.780–0.800) after only 5 epochs. CheXNet (Stanford 2017) reported 0.841 with full training.

---

## 🔥 Grad-CAM Explainability

Gradient-weighted Class Activation Mapping (Grad-CAM) generates heatmaps showing which regions of the X-ray influenced each prediction. Red/yellow areas indicate high model attention.

**Why this matters:** Explainable AI is legally required under EU AI Act (2025) and recommended by FDA for medical AI systems. Clinicians need to verify AI decisions — Grad-CAM makes this possible.

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Model Training | PyTorch + torchvision |
| Architecture | EfficientNet-B3 + Custom SE Block |
| Explainability | Grad-CAM (custom implementation) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containerisation | Docker |
| API Deployment | HuggingFace Spaces |
| Frontend Deployment | Streamlit Community Cloud |
| Dataset | NIH ChestX-ray14 (112,120 images) |
| Training Platform | Kaggle (Tesla T4 GPU) |

---

## 📁 Project Structure

```
chest-xray-classifier/
├── api.py              # FastAPI backend — /health, /model-info, /predict
├── app.py              # Streamlit medical dashboard frontend
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container for HuggingFace deployment
├── model/
│   └── best_model.pth  # Trained weights (downloaded at runtime from GitHub Releases)
└── README.md
```

---

## ⚙️ Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/SaugatDeo/chest-xray-classifier.git
cd chest-xray-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the FastAPI backend
```bash
uvicorn api:app --reload
```

API will be live at `http://127.0.0.1:8000`  
Interactive docs at `http://127.0.0.1:8000/docs`

### 4. Start the Streamlit frontend
```bash
streamlit run app.py
```

Open `http://localhost:8501`, upload a chest X-ray and get predictions.

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | API status and device info |
| `/model-info` | GET | Architecture, dataset, and performance metrics |
| `/predict` | POST | Upload X-ray → returns 14 disease probabilities + Grad-CAM |

### Sample /predict Response

```json
{
  "predictions": {
    "Atelectasis": 0.0114,
    "Cardiomegaly": 0.0023,
    "Pneumonia": 0.4290,
    "Edema": 0.4960
  },
  "top_diagnosis": "Edema",
  "confidence": 0.496,
  "gradcam_image": "<base64_encoded_heatmap>"
}
```

---

## 🏥 Disease Classes (14)

Atelectasis · Cardiomegaly · Consolidation · Edema · Effusion · Emphysema · Fibrosis · Hernia · Infiltration · Mass · Nodule · Pleural Thickening · Pneumonia · Pneumothorax

---

## ⚠️ Medical Disclaimer

This tool is intended for **research and educational purposes only**. It is not a substitute for professional medical diagnosis. Always consult a qualified radiologist or physician for clinical decisions. Model performance may vary across patient populations and imaging equipment.

---

## 👤 Author

**Saugat Deo**  
B.Tech Electronics & Instrumentation Engineering — NIT Rourkela (First Class, CGPA 7.30)  
Research background: Computer vision, deep learning, gait analysis, medical AI

[GitHub](https://github.com/SaugatDeo) · [LinkedIn](https://linkedin.com/in/saugat-deo-16432b228)
