import cv2
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
  
  
  
  
# import cv2
# import os

# video_name = "cctv.mp4" # or any other extension like .avi etc
# vidcap = cv2.VideoCapture(video_name)
# success,image = vidcap.read()
# count = 0
# clips_count =0
# while success:
#   if not os.path.exists("cctv_test/images%d" % clips_count):
#     os.makedirs("cctv_test/images%d" % clips_count)
#   cv2.imwrite("cctv_test/images%d/frame%d.jpg" % (clips_count,count), image)    # save frame as JPEG file      
#   success,image = vidcap.read()
#   print('Read a new frame: ', success)
#   count += 1
#   if count % 100==0:
#     clips_count+=1