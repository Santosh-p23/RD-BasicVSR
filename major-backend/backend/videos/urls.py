from django.urls import path
from .api import process_video

urlpatterns = [
    path('api/process_video/',process_video, name='process_video'),
]