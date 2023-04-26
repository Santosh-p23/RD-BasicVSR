import cv2
import os

width = 320
height = 180

fourcc = cv2.VideoWriter_fourcc('H','2','6','4')
# fourcc code for MP4
fps = 24 # number of frames per second
frame_size = (width, height) # frame size
out = cv2.VideoWriter('output2828.mp4', fourcc, fps,frame_size, isColor=True)


folder_path = 'cctv_test/images' # replace with the path to the folder containing the images
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