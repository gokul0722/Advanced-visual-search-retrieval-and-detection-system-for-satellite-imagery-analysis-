import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# EuroSAT 10 classes
class_names = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load trained model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Ask image name
image_path = input("Enter image name (example: forest.jpg): ")

image = Image.open(image_path)
image = transform(image)
image = image.unsqueeze(0)

with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

print("Predicted Class:", class_names[predicted.item()])