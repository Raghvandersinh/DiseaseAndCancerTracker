from django.shortcuts import render
from .forms import LungCancerForm, HeartDiseaseForm
from pathlib import Path
import torch
import joblib
import pandas as pd
import os
import sys

# Add the Models directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Models.lungcancermodel import LungCancerClassifier
from Models.heartdiseasetrackermodel import HeartDiseaseClassification

def home(request):
    return render(request, 'home.html')

def HeartDiseaseTracker(request):
    result = None
    form = HeartDiseaseForm()  # Initialize the form here so it's always available

    if request.method == 'POST':
        form = HeartDiseaseForm(request.POST)

        if form.is_valid():
            try:
                # Extract and process the form data
                input_data = [
                    float(form.cleaned_data['age']),
                    int(form.cleaned_data['sex']),
                    int(form.cleaned_data['cp']),
                    float(form.cleaned_data['trestbps']),
                    float(form.cleaned_data['chol']),
                    int(form.cleaned_data['fbs']),
                    int(form.cleaned_data['restecg']),
                    float(form.cleaned_data['thalach']),
                    int(form.cleaned_data['exang']),
                    float(form.cleaned_data['oldpeak']),
                    int(form.cleaned_data['slope']),
                    int(form.cleaned_data['ca']),
                    int(form.cleaned_data['thal'])
                ]

                # Create a pandas DataFrame with the form data
                columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
                df = pd.DataFrame([input_data], columns=columns)

                # Select only the columns that need to be scaled
                columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
                scaler_path = Path.cwd().parent / 'Models' / 'SavedModels' / 'HeartDiseaseScaler.pkl'
                scaler = joblib.load(scaler_path)

                # Apply the scaler to the selected columns
                df[columns_to_scale] = scaler.transform(df[columns_to_scale])

                # Convert the DataFrame back to a list
                input_data = df.iloc[0].tolist()


                # Load the model and make the prediction
                model = HeartDiseaseClassification()
                model_path = Path.cwd().parent / 'Models' / 'SavedModels' / 'HeartDiseaseModel.pth'
                model.load_state_dict(torch.load(model_path))

                prediction, confidence = model.predict(input_data, return_confidence=True)
                if prediction == 1:
                    result = f"Yes, you have signs of having heart disease. Consult with your doctor. Confidence: {confidence:.2f}%"
                else:
                    result = f"No, you don't have heart disease. Confidence: {confidence:.2f}%"

            except Exception as e:
                # If an error occurs during prediction or data transformation
                print(f"Error during prediction: {e}")
                result = "An error occurred while making the prediction. Please try again."

        else:
            # If form is invalid, log errors
            print(f"Form validation failed: {form.errors}")
            result = "There was an error with the form submission. Please check the input values."

    return render(request, 'HeartDiseaseTracker.html', {'form': form, 'result': result})

    
def LungCancerTracker(request):
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
            model = LungCancerClassifier()
            model_path = base_path / 'Models' / 'SavedModels' / 'lung_cancer_model.pth'
            model.load_state_dict(torch.load(model_path))
            prediction, confidence = model.predict(input_data, return_confidence=True)

            if prediction == 1:
                result = f"Yes, you have signs of having cancer. Consult with your doctor. Confidence: {confidence:.2f}%"
            else:
                result = f"No, you don't have cancer. Confidence: {confidence:.2f}%"
    else:
        form = LungCancerForm()

    return render(request, 'LungCancerTrackerModel.html', {'form': form, 'result': result})
