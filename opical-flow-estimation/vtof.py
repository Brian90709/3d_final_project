import cv2

interval = 1 # 保存时的帧数间隔
frame_count = 0 # 保存帧的索引
frame_index = 0 # 原视频的帧索引，与 interval*frame_count = frame_index 

cap = cv2.VideoCapture("/tmp2/r10922026/speedchallenge/data/train.mp4")

if cap.isOpened():
    success = True
else:
    success = False
    print("读取失败!")

while(success):
    success, frame = cap.read()
    if success is False:
        print("---> 第%d帧读取失败:" % frame_index)
        break
        
    print("---> 正在读取第%d帧:" % frame_index, success)
    if frame_index % interval == 0:
        cv2.imwrite(f'/tmp2/r10922026/opical-flow-estimation-with-RAFT/train/{frame_count}.jpg', frame)
        frame_count += 1
    frame_index += 1
