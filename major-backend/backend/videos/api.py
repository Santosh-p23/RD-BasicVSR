from rest_framework import viewsets,generics
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import Videos
from .apps import VideosConfig
from .serializers import VideoSerializer

from rest_framework.parsers import FormParser,MultiPartParser
from django.http import HttpResponse
from django.conf import settings
from .vid_ups import upsample


import cv2
import os

width = 1280
height = 720
fps = 24

@api_view(['POST'])
def process_video(request):
         
    #get the video file from the request
    video_file = request.FILES.get('video')
    
    file_path = os.path.join(VideosConfig.mediapath, video_file.name)
    with open(file_path, 'wb+') as destination:
        for chunk in video_file.chunks():
            destination.write(chunk)

    vidcap = cv2.VideoCapture(file_path)
    
    print(video_file.name)
    
    
    if not vidcap.isOpened():
            return HttpResponse("Error: could not open video capture object.")
    success,image = vidcap.read()
    count = 0
    while success:
        a = str(count).zfill(4)
        test_path=VideosConfig.mediapath+'cctv_test/images/'
        cv2.imwrite(f'{test_path}/frame{a}.jpg', image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
    
    # perform operations on the video  
    # perform vid_ups here
    
    upsample()

    
    
    
    # save the processed video to a file
    output_file = 'processed_video.mp4'
    fourcc = cv2.VideoWriter_fourcc('H','2','6','4')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    # write the processed frames to the output video
    folder_path = VideosConfig.mediapath+'flowvsr/images' # replace with the path to the folder containing the images
    frame_list = [] # list to store the frames

    # get a list of filenames in the folder
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

    # sort the filenames to ensure that the frames are in order
    filenames.sort()
    print(filenames)

    for filename in filenames:
        image_path = os.path.join(folder_path, filename)
        frame = cv2.imread(image_path)
        frame_list.append(frame)
        out.write(frame)
    out.release()
    
    dirs = [ VideosConfig.mediapath+'/flowvsr/images', VideosConfig.mediapath+'/cctv_test/images']
    for dir_path in dirs:
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    
    
    # return the processed video as a response
    with open(output_file, 'rb') as f:
        response = HttpResponse(f.read(), content_type='video/mp4')
        response['Content-Disposition'] = 'attachment; filename="processed_video.mp4"'
    return response