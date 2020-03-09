import os.path
import random
import sys

import cv2
import h5py
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform
from data.brats_dataset import BratsDataset
from data.brats_hgglgg_dataset import BratsHgglggDataset
from data.nuclei_dataset import NucleiDataset


class NucleiSplitDataset(BaseDataset):

    def __init__(self, opt):

        self.split_db = []
        for i in range(4):
            self.split_db.append(NucleiDataset(opt, i))

    def __getitem__(self, index):

        result = {}
        for k, v in enumerate(self.split_db):
            database = v
            if index >= len(database):
                index = index % len(database)

            index_value = database[index]
            result['A_' + str(k)] = index_value['A']
            result['B_' + str(k)] = index_value['B']
            # result['Seg_' + str(k)] = index_value['Seg']
            result['A_paths_' + str(k)] = index_value['A_paths']
            result['B_paths_' + str(k)] = index_value['B_paths']

        return result

    def __len__(self):
        """Return the total number of images in the dataset."""
        length = 0
        for i in self.split_db:
            if len(i) > length:
                length = len(i)

        return length

