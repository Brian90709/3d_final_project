from __future__ import division, print_function
import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from utils import flow_viz
from utils.utils import InputPadder
import yaml
import evaluate, viz
from deq import get_model
import torch.nn as nn


DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def viz(img, flo, i):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)
    
    plt.imshow(img_flo / 255.0)
    plt.savefig(f'/home/sharif/Documents/RAFT/test_vis/{i}.png')

    # clear plt
    plt.clf()
    plt.cla()

def run(args):
    # model = torch.nn.DataParallel(RAFT(args))
#     with open(os.path.join('config_folder', 'kitti.yaml')) as f:
#         config = cf.Reader(yaml.safe_load(f))
#     with open(os.path.join('config_folder', 'MaskFlownet_kitti.yaml')) as f:
#         config_model = cf.Reader(yaml.safe_load(f))
        
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     model = eval(config_model.value['network']['class'])(config)
#     checkpoint = torch.load(os.path.join('weights', '8caNov12-1532_300000.pth'))
#     model.load_state_dict(checkpoint)
#     model = model.to(device)

    DEQFlow = get_model()
    model = nn.DataParallel(DEQFlow(), device_ids=[0])
    # print("Parameter Count: %.3f M" % count_parameters(model))
    
    # if args.restore_ckpt is not None:
    model.load_state_dict(torch.load("checkpoints/deq-flow-H-kitti.pth"), strict=False)

    model.cuda()
    # model.eval()
    
    
    # model.load_state_dict(torch.load('models/raft-things.pth'))

    # model = model.module
    # model.to(DEVICE)
    model.eval()

    output_dir = Path(args.output_dir)
    images_dir = Path(args.images_dir)
    images = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))

    with torch.no_grad():
        images = sorted(images)

        for i in range(16920, len(images)-1):
            im_f1 = str(images[i])
            im_f2 = str(images[i+1])
            
            image1 = load_image(im_f1)
            image2 = load_image(im_f2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up, info = model(image1, image2, True)
                
            # 2.2 MB
            of_f_name = output_dir / f'{i}.npy' 
            np.save(of_f_name, flow_up.cpu())
            print(f'optical file {of_f_name}, shape = {flow_up.cpu().shape}')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', help="directory with your images")
    parser.add_argument('--output_dir', help="optical flow images will be stored here as .npy files")
    args = parser.parse_args()

    run(args)
