from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("LungCancerTracker/", views.LungCancerTracker, name="LungCancerTracker"),
    path("HeartDiseaseTracker/", views.HeartDiseaseTracker, name="HeartDiseaseTracker"),
    path("PneumoniaTracker/", views.PneumoniaTracker, name="PneumoniaTracker"),
    path("upload/", views.upload_image, name="upload_image"),  # Add this line
]