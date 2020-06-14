#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

import networks
from utils import download_model_if_doesnt_exist
from layers import *
import datasets
from utils import *
from kitti_utils import *
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import time
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='name of the model', default='monodepth')
parser.add_argument('--models_dir', type=str, help='directory containing the state dict of the models')
parser.add_argument('--dataset_dir', type=str, help='directory containing the dataset')
parser.add_argument('--split_dir', type=str, help='directory containing the split')
parser.add_argument('--results_dir', type=str, help='directory containing the results')
parser.add_argument('--show_nice', action='store_true')
args = parser.parse_args()

# ## Utils
def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring (fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3) 
    return buf
 
def fig2img(fig, width, height):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    img = pil.frombytes("RGB", (w, h), buf.tostring())
    img = img.resize((width, height))
    return img


# ## Load model
encoder_path = os.path.join(args.models_dir, args.model_name, "encoder.pth")
depth_decoder_path = os.path.join(args.models_dir, args.model_name, "depth.pth")

# LOADING PRETRAINED MODEL
encoder = networks.ResnetEncoder(18, False)
depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)

loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
depth_decoder.load_state_dict(loaded_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = encoder.to(device)
depth_decoder = depth_decoder.to(device)

encoder.eval()
depth_decoder.eval();

# LOADING DATASET
fpath = os.path.join(args.split_dir, "{}_files.txt")
test_filenames = readlines(fpath.format("train"))
img_ext = '.png'

test_dataset = datasets.UPBRAWDataset(
    args.dataset_dir, test_filenames, 256, 512,
    [0, -1, 1], 4, is_train=True, img_ext=img_ext)


# ## Run experiment
def run(idx: int):
    sample = test_dataset[idx]
    original_height, original_width = sample['color', 0, 0].shape[1:]
    input_image = sample['color', 0, 0].numpy().transpose(1, 2, 0)
    input_image_pytorch = sample['color', 0, 0].unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = encoder(input_image_pytorch)
        outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]
    
    disp_resized = torch.nn.functional.interpolate(disp,
    (original_height, original_width), mode="bilinear", align_corners=False)

    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    

    figure = plt.figure(figsize=(10, 10))
    plt.subplot(211)
    plt.imshow(input_image)
    plt.title("Input", fontsize=22)
    plt.axis('off')

    plt.subplot(212)
    plt.imshow(disp_resized_np, cmap='magma', vmax=vmax)
    plt.title("Disparity prediction", fontsize=22)
    plt.axis('off');
    img = fig2img(figure, height=2*original_height, width=original_width)    
    img.save(os.path.join(args.results_dir, str(idx) + ".png"))
    plt.close(figure)


if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")

    idx = np.random.randint(0, len(test_dataset) -1, 100)
    for i in tqdm(idx):
        run(i)





