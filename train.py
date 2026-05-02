import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# 1. Hyperparameters & Settings
batch_size = 32
epochs = 5
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Data Transformations (Preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Load Dataset (EuroSAT)
# Neenga manual-ah download pannala na 'download=True' kudunga
train_dataset = datasets.EuroSAT(root=".", download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 4. Model Setup (ResNet18)
model = models.resnet18(weights='IMAGENET1K_V1')
# EuroSAT-la 10 categories iruku (River, Forest, Highway, etc.)
model.fc = nn.Linear(model.fc.in_features, 10) 
model = model.to(device)

# 5. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 6. Training Loop
print(f"Training started on {device}...")
model.train()

for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 7. Save the Model
torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth!")