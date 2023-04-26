from rest_framework import serializers
from .models import Videos

class VideoSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = Videos
        fields = ('id','videofile')