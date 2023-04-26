import cv2
import os
from vid_ups import *

video_name = "cctv.mp4" # or any other extension like .avi etc
vidcap = cv2.VideoCapture(video_name)
success,image = vidcap.read()
count = 0
while success:
  a = str(count).zfill(4)
  cv2.imwrite(f'cctv_test/images/frame{a}.jpg', image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
  

def upscale():
  
    test_dataset = ValImageDataset(VAL_LR_PATH,100)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    model = BasicVSRNet()
    model = model.to(DEVICE)

    load_model_opt(
        MODEL_PATH,#give here model path
        model,
        )

    #run validate function to generate images from lr images       
    validate(model,test_dataloader)   

width = 1280
height = 720

fourcc = cv2.VideoWriter_fourcc(*'mp4v') # fourcc code for MP4
fps = 24 # number of frames per second
frame_size = (width, height) # frame size
out = cv2.VideoWriter('output.mp4', fourcc, fps,frame_size, isColor=True)


folder_path = 'flowvsr/images' # replace with the path to the folder containing the images
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