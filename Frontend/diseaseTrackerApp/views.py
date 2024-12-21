from django.shortcuts import render, HttpResponse
from .models import LungCancerTrackerModel
# Create your views here.
def home(request):
    return render(request, "home.html")
def LungCancerTracker(request):
    items = LungCancerTrackerModel.objects.all()
    return render(request, "LungCancerTrackerModel.html", {'LungCancerTrackerModel': items})