import io
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# ── Disease classes ──────────────────────────────────────────────────────────
DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
    'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
    'Pneumonia', 'Pneumothorax'
]

# ── Model architecture (must match training) ─────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChestXrayModel(nn.Module):
    def __init__(self, num_classes=14, dropout=0.4):
        super(ChestXrayModel, self).__init__()
        efficientnet = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
        self.backbone = efficientnet.features
        self.avgpool = efficientnet.avgpool
        backbone_out = 1536
        self.se_block = SEBlock(channels=backbone_out, reduction=16)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.75),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.se_block(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


# ── Load model ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChestXrayModel(num_classes=14, dropout=0.4)
checkpoint = torch.load("model/best_model.pth", map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)
print(f"Model loaded on {device}")

# ── Image transform ──────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── Grad-CAM ─────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer = model.backbone[-1]
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, class_idx):
        self.model.eval()
        output = self.model(image_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224),
                            mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


gradcam = GradCAM(model)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Chest X-Ray Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def image_to_base64(img_array):
    _, buffer = cv2.imencode('.png', img_array)
    return base64.b64encode(buffer).decode('utf-8')


@app.get("/health")
def health():
    return {"status": "ok", "model": "EfficientNet-B3-ChestXray", "device": str(device)}


@app.get("/model-info")
def model_info():
    return {
        "architecture": "EfficientNet-B3 + Custom SE Attention Block",
        "dataset": "NIH ChestX-ray14",
        "total_images": 112120,
        "train_images": 86524,
        "test_images": 25596,
        "num_classes": 14,
        "disease_classes": DISEASE_CLASSES,
        "mean_auc_roc": 0.7933,
        "training_epochs": 5
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image_resized = image.resize((224, 224))
    image_np = np.array(image_resized)

    # Preprocess
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.sigmoid(outputs)[0].cpu().numpy()

    # Build predictions dict
    predictions = {
        cls: round(float(prob), 4)
        for cls, prob in zip(DISEASE_CLASSES, probs)
    }

    # Top diagnosis
    top_idx = int(np.argmax(probs))
    top_disease = DISEASE_CLASSES[top_idx]
    top_confidence = round(float(probs[top_idx]), 4)

    # Grad-CAM for top prediction
    image_tensor_grad = transform(image).unsqueeze(0).to(device)
    image_tensor_grad.requires_grad = True
    cam = gradcam.generate(image_tensor_grad, top_idx)

    # Overlay heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * image_np / 255.0 + 0.4 * heatmap / 255.0)
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    return {
        "predictions": predictions,
        "top_diagnosis": top_disease,
        "confidence": top_confidence,
        "gradcam_image": image_to_base64(overlay_bgr)
    }