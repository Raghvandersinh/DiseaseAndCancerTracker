from django.shortcuts import render, redirect
from .models import LungCancerTrackerModel
from .forms import LungCancerTrackerForm

def home(request):
    return render(request, "home.html")

def LungCancerTracker(request):
    if request.method == 'POST':
        form = LungCancerTrackerForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('LungCancerTracker')
    else:
        form = LungCancerTrackerForm()
    items = LungCancerTrackerModel.objects.all()
    return render(request, "LungCancerTrackerModel.html", {'form': form, 'LungCancerTrackerModel': items})