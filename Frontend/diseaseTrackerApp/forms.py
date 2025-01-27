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
    cp = forms.ChoiceField(choices=[(0, 'Asymptomatic'), (1, 'Non-anginal Pain'), (2, 'Atypical Angina'), (3, 'Typical Angina')], label='Chest Pain Type')
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
                    widget=forms.NumberInput(attrs={'min': 0, 'max': 6.2, 'step':0.1})  # min/max for HTML input
)
    slope = forms.ChoiceField(choices=[(0, 'Upsloping'), (1, 'Flat'), (2, 'Downsloping')], label='Slope of the Peak Exercise ST Segment')
    ca = forms.ChoiceField(choices=[(0, '0'), (1, '1'), (2, '2'), (3, '3')], label='Number of Major Vessels Colored by Flourosopy')
    thal = forms.ChoiceField(choices=[(0, 'Normal'), (1, 'Fixed Defect'), (2, 'Reversable Defect')], label='Thalassemia')

class PneumoniaForm(forms.Form):
    Xray = forms.ImageField(label='Upload X-ray')
    
class BreastCancerForm(forms.Form):
    radius_mean = forms.FloatField(label='Radius Mean', validators=[MinValueValidator(6.981), MaxValueValidator(28.11)],
                    widget=forms.NumberInput(attrs={'min': 6.981, 'max': 28.11})  # min/max for HTML input
)
    texture_mean = forms.FloatField(label='Texture Mean', validators=[MinValueValidator(9.71), MaxValueValidator(39.28)],
                    widget=forms.NumberInput(attrs={'min': 9.71, 'max': 39.28})  # min/max for HTML input
)
    perimeter_mean = forms.FloatField(label='Perimeter Mean', validators=[MinValueValidator(43.79), MaxValueValidator(188.5)],
                    widget=forms.NumberInput(attrs={'min': 43.79, 'max': 188.5})  # min/max for HTML input
)
    area_mean = forms.FloatField(label='Area Mean', validators=[MinValueValidator(143.5), MaxValueValidator(2501)],
                    widget=forms.NumberInput(attrs={'min': 143.5, 'max': 2501})  # min/max for HTML input
)
    smoothness_mean = forms.FloatField(label='Smoothness Mean', validators=[MinValueValidator(0.053), MaxValueValidator(0.163)],
                    widget=forms.NumberInput(attrs={'min': 0.053, 'max': 0.163})  # min/max for HTML input
)
    compactness_mean = forms.FloatField(label='Compactness Mean', validators=[MinValueValidator(0.01938), MaxValueValidator(0.3454)],
                    widget=forms.NumberInput(attrs={'min': 0.01938, 'max': 0.3454})  # min/max for HTML input
)
    concavity_mean = forms.FloatField(label='Concavity Mean', validators=[MinValueValidator(0), MaxValueValidator(0.4268)],
                    widget=forms.NumberInput(attrs={'min': 0, 'max': 0.4268})  # min/max for HTML input
)
    concave_points_mean = forms.FloatField(label='Concave Points Mean', validators=[MinValueValidator(0), MaxValueValidator(0.2012)],
                    widget=forms.NumberInput(attrs={'min': 0, 'max': 0.2012})  # min/max for HTML input
)
    symmetry_mean = forms.FloatField(label='Symmetry Mean', validators=[MinValueValidator(0.106), MaxValueValidator(0.304)], 
                     widget=forms.NumberInput(attrs={'min': 0.106, 'max': 0.304})  # min/max for HTML input
    )
    fractal_dimension_mean = forms.FloatField(label='Fractal Dimension Mean', validators=[MinValueValidator(0.05), MaxValueValidator(0.09744)],
                    widget=forms.NumberInput(attrs={'min': 0.05, 'max': 0.09744})  # min/max for HTML input 
    )
    radius_se = forms.FloatField(label='Radius SE', validators=[MinValueValidator(0.1115), MaxValueValidator(2.873)],
                    widget=forms.NumberInput(attrs={'min': 0.1115, 'max': 2.873})  # min/max for HTML input
    )
    texture_se = forms.FloatField(label='Texture SE', validators=[MinValueValidator(0.3602), MaxValueValidator(4.885)],
                    widget=forms.NumberInput(attrs={'min': 0.3602, 'max': 4.885})  # min/max for HTML input
    )
    perimeter_se = forms.FloatField(label='Perimeter SE', validators=[MinValueValidator(0.757, 21.98)],
                    widget=forms.NumberInput(attrs={'min': 0.757, 'max': 21.98})  # min/max for HTML input
    )
    area_se = forms.FloatField(label='Area SE', validators=[MinValueValidator(6.802), MaxValueValidator(542.2)],
                    widget=forms.NumberInput(attrs={'min': 6.802, 'max': 542.2})  # min/max for HTML input
    )
    smoothness_se = forms.FloatField(label='Smoothness SE', validators=[MinValueValidator(0.002252), MaxValueValidator(0.03113)],
                    widget=forms.NumberInput(attrs={'min': 0.002252, 'max': 0.03113})  # min/max for HTML input
    )
    compactness_se = forms.FloatField(label='Compactness SE', validators=[MinValueValidator(0.002252), MaxValueValidator(0.1354)],
                    widget=forms.NumberInput(attrs={'min': 0.002252, 'max': 0.1354})  # min/max for HTML input
    )
    concavity_se = forms.FloatField(label='Concavity SE', validators=[MinValueValidator(0), MaxValueValidator(0.396)],
                    widget=forms.NumberInput(attrs={'min': 0, 'max': 0.396})  # min/max for HTML input
    )
    concave_points_se = forms.FloatField(label='Concave Points SE', validators=[MinValueValidator(0), MaxValueValidator(0.05279)],
                    widget=forms.NumberInput(attrs={'min': 0, 'max': 0.05279})  # min/max for HTML input
    )
    symmetry_se = forms.FloatField(label='Symmetry SE', validators=[MinValueValidator(0.008772), MaxValueValidator(0.07895)],
                    widget=forms.NumberInput(attrs={'min': 0.008772, 'max': 0.07895})  # min/max for HTML input
    )
    fractal_dimension_se = forms.FloatField(label='Fractal Dimension SE', validators=[MinValueValidator(0.001713), MaxValueValidator(0.02984)],
                    widget=forms.NumberInput(attrs={'min': 0.001713, 'max': 0.02984})  # min/max for HTML input
    )
    radius_worst = forms.FloatField(label='Radius Worst', validators=[MinValueValidator(7.93), MaxValueValidator(36.04)],
                    widget=forms.NumberInput(attrs={'min': 7.93, 'max': 36.04})  # min/max for HTML input
    )
    texture_worst = forms.FloatField(label='Texture Worst', validators=[MinValueValidator(12.02), MaxValueValidator(49.54)],
                    widget=forms.NumberInput(attrs={'min': 12.02, 'max': 49.54})  # min/max for HTML input
    )
    perimeter_worst = forms.FloatField(label='Perimeter Worst', validators=[MinValueValidator(50.41), MaxValueValidator(251.2)],
                    widget=forms.NumberInput(attrs={'min': 50.41, 'max': 251.2})  # min/max for HTML input
    )
    area_worst = forms.FloatField(label='Area Worst', validators=[MinValueValidator(185.2), MaxValueValidator(4254)],
                    widget=forms.NumberInput(attrs={'min': 185.2, 'max': 4254})  # min/max for HTML input
    )
    smoothness_worst = forms.FloatField(label='Smoothness Worst', validators=[MinValueValidator(0.07117), MaxValueValidator(0.2226)],
                    widget=forms.NumberInput(attrs={'min': 0.07117, 'max': 0.2226})  # min/max for HTML input
    )
    compactness_worst = forms.FloatField(label='Compactness Worst', validators=[MinValueValidator(0.02729), MaxValueValidator(1.058)],
                    widget=forms.NumberInput(attrs={'min': 0.02729, 'max': 1.058})  # min/max for HTML input
    )
    concavity_worst = forms.FloatField(label='Concavity Worst', validators=[MinValueValidator(0), MaxValueValidator(1.252)],
                    widget=forms.NumberInput(attrs={'min': 0, 'max': 1.252})  # min/max for HTML input
    )
    concave_points_worst = forms.FloatField(label='Concave Points Worst', validators=[MinValueValidator(0), MaxValueValidator(0.291)],
                    widget=forms.NumberInput(attrs={'min': 0, 'max': 0.291})  # min/max for HTML input
    )
    symmetry_worst = forms.FloatField(label='Symmetry Worst', validators=[MinValueValidator(0.1565), MaxValueValidator(0.6638)],
                    widget=forms.NumberInput(attrs={'min': 0.1565, 'max': 0.6638})  # min/max for HTML input
    )
    fractal_dimension_worst = forms.FloatField(label='Fractal Dimension Worst', validators=[MinValueValidator(0.05504), MaxValueValidator(0.2075)],
                    widget=forms.NumberInput(attrs={'min': 0.05504, 'max': 0.2075})  # min/max for HTML input
    )
    
    
                    