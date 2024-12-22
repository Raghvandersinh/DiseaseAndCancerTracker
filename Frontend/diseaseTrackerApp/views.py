import os
from pathlib import Path
import torch
import torch.nn as nn
from django.shortcuts import render, redirect
from .models import LungCancerTrackerModel
from .forms import LungCancerTrackerForm
import joblib

class LungCancerClassifier(nn.Module):
    def __init__(self, input_size):
        super(LungCancerClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)

def load_model():
    model = LungCancerClassifier(input_size=15)  # Adjust input_size based on your model
    base_path = Path(__file__).resolve().parent.parent.parent
    model_path = base_path / 'Models' / 'SavedModels' / 'lung_cancer_model.pth'
    print(f"Model path: {model_path}")  # Print the model path for debugging
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_scaler():
    base_path = Path(__file__).resolve().parent.parent.parent
    scaler_path = base_path / 'Models' / 'SavedModels' / 'age_scaler.pkl'
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    scaler = joblib.load(scaler_path)
    return scaler

def predict_lung_cancer(data):
    model = load_model()
    scaler = load_scaler()
    data[0] = scaler.transform([[data[0]]])[0][0]  # Apply scaler to the age field
    data_tensor = torch.tensor(data, dtype=torch.float32)
    with torch.no_grad():
        output = model(data_tensor.unsqueeze(0))
        prediction = (output > 0.5).float().item()
    return prediction

def home(request):
    return render(request, "home.html")

def LungCancerTracker(request):
    if request.method == 'POST':
        form = LungCancerTrackerForm(request.POST)
        if form.is_valid():
            form.save()
            data = list(form.cleaned_data.values())
            prediction = predict_lung_cancer(data)
            result = "Positive" if prediction == 1.0 else "Negative"
            return render(request, "LungCancerTrackerModel.html", {'form': form, 'result': result})
    else:
        form = LungCancerTrackerForm()
    items = LungCancerTrackerModel.objects.all()
    return render(request, "LungCancerTrackerModel.html", {'form': form, 'LungCancerTrackerModel': items})