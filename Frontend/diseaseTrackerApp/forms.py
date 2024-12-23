# filepath: /c:/MachineLearning/LungCancerTracker/Frontend/diseaseTrackerApp/forms.py
from django import forms

class LungCancerForm(forms.Form):
    gender = forms.ChoiceField(choices=[(1, 'Male'), (0, 'Female')], label='Gender')
    age = forms.FloatField(label='Age')
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