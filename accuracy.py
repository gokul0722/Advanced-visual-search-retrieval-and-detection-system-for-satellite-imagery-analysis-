import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

dataset_path = r"C:\Users\Sivaranjani P\EuroSAT"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(dataset_path, transform=transform)

from torch.utils.data import Subset
dataset = Subset(dataset, range(5000))

loader = DataLoader(dataset, batch_size=32)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("model.pth"))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Model Accuracy: {accuracy:.2f}%")