import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from tkinter import Label

# Classes

# Model Load
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("model.pth"))
model.eval()

from torchvision import datasets

dataset_path = r"C:\Users\Sivaranjani P\EuroSAT"
temp_dataset = datasets.ImageFolder(dataset_path)
class_names = temp_dataset.classes

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def predict_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    image = Image.open(file_path)
    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    result_label.config(text="Prediction: " + class_names[predicted.item()])

# GUI
root = tk.Tk()
root.title("Satellite Image Classifier")

btn = tk.Button(root, text="Select Image", command=predict_image)
btn.pack(pady=20)

result_label = Label(root, text="Prediction: ")
result_label.pack(pady=20)

root.mainloop()