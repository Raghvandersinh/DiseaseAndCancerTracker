from django.shortcuts import render
from .forms import LungCancerForm
import torch
import os
import sys
from pathlib import Path
from asgiref.sync import sync_to_async
import joblib
import pandas as pd

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
            input_data = [
                int(form.cleaned_data['gender']),
                int(form.cleaned_data['age']),
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
            
            # Convert input_data[1] to a pandas DataFrame with the correct column name
            age_df = pd.DataFrame([[input_data[1]]], columns=['AGE'])

            # Transform the age value using the scaler
            transformed_age = scaler.transform(age_df)

            # Update input_data[1] with the transformed value
            input_data[1] = transformed_age[0][0]
            
            # Make the prediction
            result = await sync_to_async(predict_cancer)(input_data)
    else:
        form = LungCancerForm()
    
    return render(request, 'LungCancerTrackerModel.html', {'form': form, 'result': result})

def predict_cancer(input_data):
    model = LungCancerClassifier()
    base_path = Path(__file__).resolve().parent.parent.parent
    model_path = base_path / 'Models' / 'SavedModels' / 'lung_cancer_model.pth'
    model.load_state_dict(torch.load(model_path))
    prediction, confidence = model.predict(input_data, return_confidence=True)
    
    # Set result message based on prediction
    if prediction == 1:
        return f"Yes, you have signs of having cancer. Consult with your doctor. Confidence: {confidence:.2f}%"
    else:
        return f"No, you don't have cancer. Confidence: {confidence:.2f}%"
