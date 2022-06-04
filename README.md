# 3d_final_project

## How to use

You need to download the speedchallenge dataset and  inference optical flow maps first.

Before inference, you need to make the video into frames.
```shell
git clone https://github.com/commaai/speedchallenge.git
python cvt.py
cd opical-flow-estimation
```
and read the README.md.

After inferenced the optical flow maps, train the following model.

### PWC-Net
```shell
python trainPWC.py
```

### LiteFlowNet
```shell
python trainLite.py
```

### FlowNet2
```shell
python trainFlow.py
```

### DEQ Flow
```shell
python trainDEQ.py
```

### RAFT
Tutorial in https://github.com/Brian90709/3d_final_project/blob/main/vehicle-speed-estimation/README.txt

### OpenCV Dense Optical Flow
of_opencv.ipynb
