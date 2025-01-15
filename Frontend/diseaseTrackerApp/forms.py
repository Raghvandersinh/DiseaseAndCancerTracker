# filepath: /c:/MachineLearning/LungCancerTracker/Frontend/diseaseTrackerApp/forms.py
from django import forms
from django.core.validators import MinValueValidator, MaxValueValidator

class LungCancerForm(forms.Form):
    gender = forms.ChoiceField(choices=[(1, 'Male'), (0, 'Female')], label='Gender')
    age = forms.FloatField(label='Age', validators=[MinValueValidator(0), MaxValueValidator(100)])
    smoking = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')], label='Smoking')
    yellow_fingers = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')], label='Yellow Fingers')
    anxiety = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')], label='Anxiety')
    peer_pressure = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')], label='Peer Pressure')
    chronic_disease = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')], label='Chronic Disease')
    fatigue = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')], label='Fatigue')
    allergy = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')], label='Allergy')
    wheezing = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')], label='Wheezing')
    alcohol_consuming = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')], label='Alcohol Consuming')
    coughing = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')], label='Coughing')
    shortness_of_breath = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')], label='Shortness of Breath')
    swallowing_difficulty = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')], label='Swallowing Difficulty')
    chest_pain = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')], label='Chest Pain')
    
class HeartDiseaseForm(forms.Form):
    age = forms.FloatField(label='Age', validators=[MinValueValidator(0), MaxValueValidator(100)],
                           widget=forms.NumberInput(attrs={'min': 0, 'max': 100})  # min/max for HTML input
)
    sex = forms.ChoiceField(choices=[(1,'Male'), (0, 'Female')], label='Gender')
    cp = forms.ChoiceField(choices=[(0, 'Typical Angina'), (1, 'Atypical Angina'), (2, 'Non-anginal Pain'), (3, 'Asymptomatic')], label='Chest Pain Type')
    trestbps = forms.FloatField(label='Resting Blood Pressure', validators=[MinValueValidator(94), MaxValueValidator(200)],
                                       widget=forms.NumberInput(attrs={'min': 94, 'max': 200})  # min/max for HTML input
)
    chol = forms.FloatField(label='Cholestoral in mg/dl', validators=[MinValueValidator(126), MaxValueValidator(564)],
                                    widget=forms.NumberInput(attrs={'min': 126, 'max': 564})  # min/max for HTML input
)
    fbs = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')], label='Fasting Blood Sugar > 120 mg/dl')   
    restecg = forms.ChoiceField(choices=[(0, 'Normal'), (1, 'ST-T Wave Abnormality'), (2, 'Left Ventricular Hypertrophy')], label='Resting Electrocardiographic Results')
    thalach = forms.FloatField(label='Maximum Heart Rate Achieved', validators=[MinValueValidator(71), MaxValueValidator(202)],
                                    widget=forms.NumberInput(attrs={'min': 71, 'max': 202})  # min/max for HTML input
)
    exang = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')], label='Exercise Induced Angina')
    oldpeak = forms.FloatField(label='ST Depression Induced by Exercise Relative to Rest', validators=[MinValueValidator(0), MaxValueValidator(6.2)],
                                     widget=forms.NumberInput(attrs={'min': 0, 'max': 6.2})  # min/max for HTML input
)
    slope = forms.ChoiceField(choices=[(0, 'Upsloping'), (1, 'Flat'), (2, 'Downsloping')], label='Slope of the Peak Exercise ST Segment')
    ca = forms.ChoiceField(choices=[(0, '0'), (1, '1'), (2, '2'), (3, '3')], label='Number of Major Vessels Colored by Flourosopy')
    thal = forms.ChoiceField(choices=[(0, 'Normal'), (1, 'Fixed Defect'), (2, 'Reversable Defect')], label='Thalassemia')
    