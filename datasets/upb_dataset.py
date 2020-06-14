from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from datasets.mono_dataset import MonoDataset


class UPBDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(UPBDataset, self).__init__(*args, **kwargs)
        self.K = np.array([[0.61, 0, 0.5, 0],   # width
                           [0, 1.22, 0.5, 0],   # height
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.full_res_shape = (640, 320)

    def check_depth(self):
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class UPBRAWDataset(UPBDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(UPBRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        image_path = os.path.join(self.data_path, folder, str(frame_index) + ".png")
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        return None

#
# if __name__ == "__main__":
#     import pandas as pd
#     with open("../splits/upb/train_files.txt") as fin:
#         filenames = fin.readlines()
#
#     print(filenames)
#     train_dataset = UPBRAWDataset(
#         data_path="../../upb_raw",
#         filenames=filenames,
#         height=128,
#         width=256,
#         frame_idxs=[0, -1, 1],
#         num_scales=4,
#         is_train=True,
#         img_ext="png")
#
#     elem = train_dataset[5]
#     print(elem.keys())
#
#     import matplotlib.pyplot as plt
#     img0 = elem[('color', -1, 0)].numpy().transpose(1, 2, 0)
#     plt.imshow(img0)
#     plt.show()
#
#     img1 = elem[('color', 0, 0)].numpy().transpose(1, 2, 0)
#     plt.imshow(img1)
#     plt.show()
#
#     img2 = elem[('color', 1, 0)].numpy().transpose(1, 2, 0)
#     plt.imshow(img2)
#     plt.show()
