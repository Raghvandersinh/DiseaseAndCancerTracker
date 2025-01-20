# -*- coding: utf-8 -*-
"""PneumoniaTracker.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tjixCod9PITRt_8XNuLHq5FsFyTPDd_D
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from matplotlib.image import imread
from matplotlib import pyplot as plt
import kaggle 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Models.HelperFunction import helperFunctions as hp
import torchvision.models as models 
from PIL import Image
import timm


folder_file_path = Path.cwd()/'dataset'/'chest_xray'
folder_location = Path.cwd()/'dataset'

if folder_file_path.exists():
    print(f"Folder already exists at: {folder_file_path}")
else:
    kaggle.api.dataset_download_files("paultimothymooney/chest-xray-pneumonia", path=folder_location, unzip=True)

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"


train_transforms = transforms.Compose([
    transforms.RandomRotation(20),  # Randomly rotate the image within a range of (-20, 20) degrees
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally with 50% probability
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Randomly change the brightness, contrast, saturation, and hue
    transforms.RandomApply([transforms.RandomAffine(0, translate=(0.1, 0.1))], p=0.5),  # Randomly apply affine transformations with translation
    transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.2)], p=0.5),  # Randomly apply perspective transformations
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])
test_val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_path = Path.cwd()/'dataset'/'chest_xray'/"chest_xray"/'train'
test_path = Path.cwd()/'dataset'/'chest_xray'/"chest_xray"/'test'
val_path = Path.cwd()/'dataset'/'chest_xray'/"chest_xray"/'val'

train_dataset = datasets.ImageFolder(root=train_path, transform=train_transforms)
test_dataset = datasets.ImageFolder(root=test_path, transform=test_val_transforms)
val_dataset = datasets.ImageFolder(root=val_path, transform=test_val_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

plt.figure(figsize=(10, 10))
plt.imshow(imread(f"{Path.cwd()}/dataset/chest_xray/chest_xray/train/NORMAL/IM-0115-0001.jpeg"))
plt.imshow(imread(f"{Path.cwd()}/dataset/chest_xray/chest_xray/train/PNEUMONIA/person1_bacteria_1.jpeg"))
plt.imshow(imread(f"{Path.cwd()}/dataset/chest_xray/chest_xray/train/PNEUMONIA/person80_virus_150.jpeg"))
print("Hello")

plt.show()
class XrayModel(nn.Module):
    def __init__(self, num_classes=3):  # Assuming binary classification (Normal/Pneumonia)
        super(XrayModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(32, 256),  # Adjust hidden size as needed
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Add dropout for regularization
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
print(device)
model = models.resnet34(weights = models.ResNet34_Weights.IMAGENET1K_V1).to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.001*0.1)

def train_and_eval():
    trained_model, _ = hp.train_and_evaluate_2d(model, train_dataloader, test_dataloader, loss, optimizer,scheduler,device, 5, 1)
    model_save_path = Path.cwd()/'Models'/'SavedModels'/'PneumoniaTrackerModel.pth'
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model saved at: {model_save_path}")

    
if __name__ == "__main__":
    train_and_eval()

