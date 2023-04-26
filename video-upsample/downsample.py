import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Open the video file
cap = cv2.VideoCapture('test_video.mp4')

# Get the video parameters
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set the target frame size for downscaling
target_width = frame_width // 4
target_height = frame_height // 4

# Create a VideoWriter object for writing the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # fourcc code for MP4
out = cv2.VideoWriter('cctv.mp4', fourcc, fps, (target_width, target_height), isColor=True)

# Define the downsampling transform using torchvision
downsampler = T.Resize((target_height, target_width), interpolation= Image.BICUBIC)

# Loop through the frames of the video
for i in range(total_frames):
    ret, frame = cap.read()
    if ret:
        # Convert the frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame from numpy to PIL
        frame = Image.fromarray(frame)

        # Downscale the frame using the transform
        downscaled_frame = downsampler(frame)

        # Convert the frame back to numpy
        downscaled_frame = np.array(downscaled_frame)

        # Convert the frame back to BGR
        downscaled_frame = cv2.cvtColor(downscaled_frame, cv2.COLOR_RGB2BGR)

        # Write the downscaled frame to the output video
        out.write(downscaled_frame)

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()