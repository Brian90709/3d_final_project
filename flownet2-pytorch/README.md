ffmpeg -i ./speedchallenge/data/train.mp4 frames/output_%02d.png

bash install.sh

Download pretrained model: https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view

python main.py --inference --model FlowNet2 --save_flow --inference_dataset ImagesFromFolder --inference_dataset_root ../speedchallenge/frames/ --resume ./checkpoints/FlowNet2_checkpoint.pth.tar --save ./output --number_gpus 1


### How to read the optical file .flo
import numpy as np

f = open('flow10.flo', 'rb')

x = np.fromfile(f, np.int32, count=1) # not sure what this gives
w = np.fromfile(f, np.int32, count=1) # width
h = np.fromfile(f, np.int32, count=1) # height

data = np.fromfile(f, np.float32) # vector 

data_2D = np.reshape(data, (2,h,w))
