from django.shortcuts import render
from .forms import LungCancerForm
import torch
import os
import sys
from pathlib import Path
from asgiref.sync import sync_to_async
import joblib

# Add the directory containing the Models module to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Models.lungcancermodel import LungCancerClassifier

def home(request):
    return render(request, 'home.html')

async def LungCancerTracker(request):
    result = None
    if request.method == 'POST':
        form = LungCancerForm(request.POST)
        if form.is_valid():
            # Extract input data from form
            input_data = [
                int(form.cleaned_data['gender']),
                int(form.cleaned_data['age']),  # Ensure age is an int
                int(form.cleaned_data['smoking']),
                int(form.cleaned_data['yellow_fingers']),
                int(form.cleaned_data['anxiety']),
                int(form.cleaned_data['peer_pressure']),
                int(form.cleaned_data['chronic_disease']),
                int(form.cleaned_data['fatigue']),
                int(form.cleaned_data['allergy']),
                int(form.cleaned_data['wheezing']),
                int(form.cleaned_data['alcohol_consuming']),
                int(form.cleaned_data['coughing']),
                int(form.cleaned_data['shortness_of_breath']),
                int(form.cleaned_data['swallowing_difficulty']),
                int(form.cleaned_data['chest_pain'])
            ]
            
            # Convert gender to numerical value
            input_data[0] = 1 if input_data[0] == 1 else 0
            
            # Load the scaler and apply it to the age feature
            base_path = Path(__file__).resolve().parent.parent.parent
            scaler_path = base_path / 'Models' / 'SavedModels' / 'age_scaler.pkl'
            scaler = joblib.load(scaler_path)
            
            # Debugging: Print the scaler and age before transformation
            print(f"Scaler: {scaler}")
            print(f"Age before transformation: {input_data[1]}")
            
            # Ensure the scaler is fitted
            if not hasattr(scaler, 'mean_'):
                print("Scaler is not fitted.")
            else:
                print(f"Scaler mean: {scaler.mean_}, var: {scaler.var_}")
            
            # Replace age with the scaler mean
            input_data[1] = scaler.mean_.item()
            
            # Debugging: Print the age after replacement
            print(f"Age after replacement: {input_data[1]}")
            
            # Debugging: Print input data
            print(f"Input data: {input_data}")
            
            # Load model and make prediction asynchronously
            result = await sync_to_async(predict_cancer)(input_data)
    else:
        form = LungCancerForm()
    
    return render(request, 'LungCancerTrackerModel.html', {'form': form, 'result': result})

def predict_cancer(input_data):
    model = LungCancerClassifier()  # Remove input_dim argument
    base_path = Path(__file__).resolve().parent.parent.parent
    model_path = base_path / 'Models' / 'SavedModels' / 'lung_cancer_model.pth'
    model.load_state_dict(torch.load(model_path))
    prediction = model.predict(input_data)
    
    # Debugging: Print model output
    print(f"Model output: {prediction}")
    
    # Set result message based on prediction
    if prediction == 1:
        return "Yes, you have signs of having cancer. Consult with your doctor."
    else:
        return "No, you don't have cancer."
