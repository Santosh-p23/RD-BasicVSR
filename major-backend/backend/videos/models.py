from django.db import models

# Create your models here.
class Videos(models.Model):
    videofile = models.FileField(upload_to="videos/", null=True, blank = True)