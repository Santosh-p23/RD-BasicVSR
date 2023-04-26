from django.apps import AppConfig
from django.conf import settings

from pathlib import Path
import os

class VideosConfig(AppConfig):
    name = 'videos'
    
    mediapath=os.path.join(settings.MEDIA_ROOT,'')
    modelpath=os.path.join(settings.MODELS,'best_model.pth')
    spynetpath= os.path.join(settings.MODELS,'spynet.pth')