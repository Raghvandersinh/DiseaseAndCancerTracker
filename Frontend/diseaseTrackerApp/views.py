from django.http import JsonResponse
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .forms import LungCancerForm, HeartDiseaseForm, PneumoniaForm, BreastCancerForm
from pathlib import Path
import torch
import joblib
import pandas as pd
import os
import sys
from django.views.decorators.csrf import csrf_exempt
import base64
from PIL import Image
from torchvision import transforms, models
import numpy as np

# Add the Models directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Models.lungcancermodel import LungCancerClassifier
from Models.heartdiseasetrackermodel import HeartDiseaseClassification
from Models.breastCancerTracker import BreastCancerClassifier

# Load the trained model
model_path = Path(__file__).resolve().parent.parent.parent / 'Models' / 'SavedModels' / 'PneumoniaTrackerModel.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def home(request):
    return render(request, 'home.html')

def PneumoniaTracker(request):
    if request.method == 'POST':
        form = PneumoniaForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['Xray']
            image = Image.open(image).convert('RGB')  # Convert image to RGB
            image = transform(image).unsqueeze(0).to(device)

            # Make the prediction
            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)
                class_names = ['Normal', 'Pneumonia']
                prediction = class_names[predicted.item()]

            # Convert the image to a base64 string for display
            image_data = image.cpu().squeeze().permute(1, 2, 0).numpy()
            image_data = np.ascontiguousarray(image_data)  # Ensure the array is C-contiguous
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            image_url = f"data:image/jpeg;base64,{image_base64}"

            return render(request, 'PneumoniaTracker.html', {'form': form, 'image_url': image_url, 'prediction': prediction})
    else:
        form = PneumoniaForm()
    return render(request, 'PneumoniaTracker.html', {'form': form})

@csrf_exempt
def upload_image(request):
    if request.method == 'POST':
        form = PneumoniaForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['Xray']
            image_url = image.url
            return JsonResponse({'image_url': image_url})
    return JsonResponse({'error': 'Invalid request'}, status=400)

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
                base_path = Path(__file__).resolve().parent.parent.parent

                scaler_path = base_path / 'Models' / 'SavedModels' / 'HeartDiseaseScaler.pkl'
                scaler = joblib.load(scaler_path)

                # Apply the scaler to the selected columns
                df[columns_to_scale] = scaler.transform(df[columns_to_scale])

                # Convert the DataFrame back to a list
                input_data = df.iloc[0].tolist()
                
                for i in [1, 2, 5, 6, 8, 10, 11, 12]:
                    input_data[i] = int(input_data[i])

                print(input_data)
                # Load the model and make the prediction
                model = HeartDiseaseClassification()
                model_path = base_path / 'Models' / 'SavedModels' / 'HeartDiseaseModel.pth'
                model.load_state_dict(torch.load(model_path, weights_only=True))

                prediction, confidence = model.predict(input_data, return_confidence=True)
                if prediction == 1:
                    result = f"Yes, you have signs of having heart disease. Consult with your doctor. Probability Of Having It: {confidence:.2f}%"
                else:
                    result = f"No, you don't have heart disease. Probability Of Having It: {confidence:.2f}%"

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
            model.load_state_dict(torch.load(model_path, weights_only=True))
            prediction, confidence = model.predict(input_data, return_confidence=True)

            if prediction == 1:
                result = f"Yes, you have signs of having cancer. Consult with your doctor. Probability Of Having It: {confidence:.2f}%"
            else:
                result = f"No, you don't have cancer. Probability Of Having It: {confidence:.2f}%"
    else:
        form = LungCancerForm()

    return render(request, 'LungCancerTrackerModel.html', {'form': form, 'result': result})

def BreastCancerTracker(request):
    result = None   
    form = BreastCancerForm()  # Initialize the form before the 'if' block
    
    if request.method == 'POST':
        form = BreastCancerForm(request.POST)  # reinitialize form with POST data
        if form.is_valid():
            # Create a dictionary with feature names as keys and the input values as floats
            input_data = {
                'radius_mean': float(form.cleaned_data['radius_mean']),
                'texture_mean': float(form.cleaned_data['texture_mean']),
                'perimeter_mean': float(form.cleaned_data['perimeter_mean']),
                'area_mean': float(form.cleaned_data['area_mean']),
                'smoothness_mean': float(form.cleaned_data['smoothness_mean']),
                'compactness_mean': float(form.cleaned_data['compactness_mean']),
                'concavity_mean': float(form.cleaned_data['concavity_mean']),
                'concave_points_mean': float(form.cleaned_data['concave_points_mean']),
                'symmetry_mean': float(form.cleaned_data['symmetry_mean']),
                'fractal_dimension_mean': float(form.cleaned_data['fractal_dimension_mean']),
                'radius_se': float(form.cleaned_data['radius_se']),
                'texture_se': float(form.cleaned_data['texture_se']),
                'perimeter_se': float(form.cleaned_data['perimeter_se']),
                'area_se': float(form.cleaned_data['area_se']),
                'smoothness_se': float(form.cleaned_data['smoothness_se']),
                'compactness_se': float(form.cleaned_data['compactness_se']),
                'concavity_se': float(form.cleaned_data['concavity_se']),
                'concave_points_se': float(form.cleaned_data['concave_points_se']),
                'symmetry_se': float(form.cleaned_data['symmetry_se']),
                'fractal_dimension_se': float(form.cleaned_data['fractal_dimension_se']),
                'radius_worst': float(form.cleaned_data['radius_worst']),
                'texture_worst': float(form.cleaned_data['texture_worst']),
                'perimeter_worst': float(form.cleaned_data['perimeter_worst']),
                'area_worst': float(form.cleaned_data['area_worst']),
                'smoothness_worst': float(form.cleaned_data['smoothness_worst']),
                'compactness_worst': float(form.cleaned_data['compactness_worst']),
                'concavity_worst': float(form.cleaned_data['concavity_worst']),
                'concave_points_worst': float(form.cleaned_data['concave_points_worst']),
                'symmetry_worst': float(form.cleaned_data['symmetry_worst']),
                'fractal_dimension_worst': float(form.cleaned_data['fractal_dimension_worst']),
            }
            
            # Replace only the underscore between 'concave' and 'points' in specific columns
            columns_to_replace = ['concave_points_mean', 'concave_points_se', 'concave_points_worst']
            input_data = {
                key.replace('concave_points', 'concave points') if key in columns_to_replace else key: value
                for key, value in input_data.items()
            }

            right_skewed_col = ['area_mean', 'texture_mean', 'perimeter_mean', 'radius_mean', 
                                'compactness_mean', 'concavity_mean', 'concave points_mean', 
                                'fractal_dimension_mean', 'area_se', 'texture_se', 'perimeter_se', 
                                'radius_se', 'compactness_se', 'concavity_se', 'concave points_se', 
                                'smoothness_se', 'symmetry_se', 'fractal_dimension_se', 'area_worst', 
                                'texture_worst', 'perimeter_worst', 'radius_worst', 'compactness_worst', 
                                'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
            
            # Convert input_data to DataFrame with the correct column names
            input_data_df = pd.DataFrame([input_data], columns=input_data.keys())
            print("Columns in input_data_df:", input_data_df.columns)

            # Check if all right_skewed_col columns exist in input_data_df
            missing_columns = [col for col in right_skewed_col if col not in input_data_df.columns]
            if missing_columns:
                print(f"Missing columns: {missing_columns}")
                return f"Error: Missing columns: {missing_columns}"

            # Load the scaler
            base_path = Path(__file__).resolve().parent.parent.parent
            scaler_path = base_path / 'Models' / 'Transformations' / 'BreastCancerScaler.pkl'   
            scaler = joblib.load(scaler_path)

            # Only transform the columns in 'right_skewed_col'
            input_data_transformed = scaler.transform(input_data_df[right_skewed_col])

            # The transformed data needs to be updated back to the original data structure if necessary
            input_data_df[right_skewed_col] = input_data_transformed

            # Load the model
            model = BreastCancerClassifier()
            model_path = base_path / 'Models' / 'SavedModels' / 'BreastCancerTracker.pth'
            model.load_state_dict(torch.load(model_path, weights_only=True))

            # Predict using the transformed data
            prediction, confidence = model.predict(input_data_df.values, return_confidence=True)
            
            if prediction == 1:
                result = f"You have Benign Tumors. Probability Of Having It: {confidence:.2f}%"
            else:
                result = f"You have Malignant Tumors. Probability Of Having It: {confidence:.2f}%"

    return render(request, 'BreastCancerTracker.html', {'form': form, 'result': result})

