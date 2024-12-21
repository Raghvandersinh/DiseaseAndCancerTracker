from django.urls import path
from.import views

urlpatterns = [
    path("", views.home, name = "home"),
    path("LungCancerTracker/", views.LungCancerTracker, name = "LungCancerTracker"),
]