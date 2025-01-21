from django.db import models

class LungCancerTrackerModel(models.Model):
    GENDER_CHOICES = [
        (1, 'Male'),
        (0, 'Female'),
    ]
    
    YES_NO_CHOICES = [
        (1, 'YES'),
        (0, 'NO'),
    ]
    
    age = models.IntegerField()
    gender = models.IntegerField(choices=GENDER_CHOICES)
    smoking = models.IntegerField(choices=YES_NO_CHOICES)
    yellow_fingers = models.IntegerField(choices=YES_NO_CHOICES)
    anxiety = models.IntegerField(choices=YES_NO_CHOICES)
    peer_pressure = models.IntegerField(choices=YES_NO_CHOICES)
    chronic_disease = models.IntegerField(choices=YES_NO_CHOICES)
    fatigue = models.IntegerField(choices=YES_NO_CHOICES)
    allergy = models.IntegerField(choices=YES_NO_CHOICES)
    wheezing = models.IntegerField(choices=YES_NO_CHOICES)
    alcohol = models.IntegerField(choices=YES_NO_CHOICES)
    coughing = models.IntegerField(choices=YES_NO_CHOICES)
    shortness_of_breath = models.IntegerField(choices=YES_NO_CHOICES)
    swallowing_difficulty = models.IntegerField(choices=YES_NO_CHOICES)
    chest_pain = models.IntegerField(choices=YES_NO_CHOICES)

class PneumoniaXray(models.Model):
    hotel_Main_Img = models.ImageField(upload_to='images/')