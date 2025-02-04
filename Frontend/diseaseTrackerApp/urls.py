from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("", views.home, name="home"),
    path("LungCancerTracker/", views.LungCancerTracker, name="LungCancerTracker"),
    path("HeartDiseaseTracker/", views.HeartDiseaseTracker, name="HeartDiseaseTracker"),
    path("PneumoniaTracker/", views.PneumoniaTracker, name="PneumoniaTracker"),
    path("BreastCancerTracker/", views.BreastCancerTracker, name="BreastCancerTracker"),
    path("upload/", views.upload_image, name="upload_image"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)