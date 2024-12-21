# filepath: /c:/MachineLearning/LungCancerTracker/Frontend/diseaseTrackerApp/forms.py
from django import forms
from .models import LungCancerTrackerModel

class LungCancerTrackerForm(forms.ModelForm):
    class Meta:
        model = LungCancerTrackerModel
        fields = '__all__'