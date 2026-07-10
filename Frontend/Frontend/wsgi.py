"""
WSGI config for Frontend project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/wsgi/
"""

import os
import sys
from django.core.wsgi import get_wsgi_application

# Add the Frontend directory to Python path
sys.path.append('/DiseaseAndCancerTracker/Frontend')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Frontend.settings')

application = get_wsgi_application()
