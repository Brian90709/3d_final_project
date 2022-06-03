# 3d_final_project

## How to use

You need to download the speedchallenge dataset and  inference optical flow maps first.

```shell
git clone https://github.com/commaai/speedchallenge.git
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
