import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model - must match the architecture used in training
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, 2)
)

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "yawn_model.pth")

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Image transform - must match validation transform (no augmentation!)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load image
image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
if not os.path.isabs(image_path):
    image_path = os.path.join(base_dir, image_path)

print(f"Testing image: {image_path}")
image = Image.open(image_path).convert("RGB")

image_tensor = transform(image).unsqueeze(0).to(device)

# Prediction
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)
    confidence, pred = torch.max(probabilities, 1)

# Classes match ImageFolder alphabetical order: "no yawn" = 0, "yawn" = 1
classes = ["no_yawn", "yawn"]

print(f"Prediction: {classes[pred.item()]}")
print(f"Confidence: {confidence.item():.4f}")
print(f"Probabilities - No Yawn: {probabilities[0][0]:.4f}, Yawn: {probabilities[0][1]:.4f}")