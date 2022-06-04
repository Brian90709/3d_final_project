import cv2
vidcap = cv2.VideoCapture('train.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frames/%05d.jpg" % count, image)     # save frame as JPG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
