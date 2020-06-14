from __future__ import absolute_import, division, print_function
import os
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

import networks
from utils import download_model_if_doesnt_exist

if __name__ == "__main__":
	model_name = "mono_640x192"
	download_model_if_doesnt_exist(model_name)