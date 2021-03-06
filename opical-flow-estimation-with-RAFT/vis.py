import sys
sys.path.append('core')
from utils import flow_viz
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import click

@click.command()
@click.option("--npy_dir", help="Full path to the directory with the numpy optical flow files")
@click.option("--output_dir")
def vis(npy_dir, output_dir):
    # npy_dir = Path(npy_dir)
    # output_dir = Path(output_dir)

    # npy_files = list(npy_dir.glob('*.npy'))

    # for i, npy_file in enumerate(npy_files):
    #     print(npy_file)
        # f = str(npy_file)
        # of = np.load(f)
        # of = torch.from_numpy(of)
        # of = of[0].permute(1,2,0).numpy()
        # of = flow_viz.flow_to_image(of)
        # img = Image.fromarray(of)
        # output_f = output_dir / npy_file.stem
        # output_f = str(output_f) + '.jpg'
        # print(output_f)
        # img.save(output_f)
    for i in range(20399):
        npy_file = npy_dir + '/' + str(i) + '.npy'
        # f = str(npy_file)
        of = np.load(npy_file)
        of = torch.from_numpy(of)
        of = of[0].permute(1,2,0).numpy()
        of = flow_viz.flow_to_image(of)
        img = Image.fromarray(of)
        # output_f = output_dir / npy_file.stem
        # output_f = str(output_f) + '.jpg'
        output_f = output_dir + '/' +str(i) + '.jpg'
        img.save(output_f)
        if i % 20 == 0: print(i)

if __name__ == '__main__':
    vis()
