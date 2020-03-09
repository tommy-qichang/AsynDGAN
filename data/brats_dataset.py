import os.path
import random
import sys

import cv2
import h5py
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform


class BratsDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt, idx=None):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if idx is None:
            h5_name = "BraTS18.h5"
        else:
            h5_name = f"BraTS18_tumor_size_{idx}.h5"
            print(f"Load: {h5_name}")
        self.is_test = True
        self.real_tumor = False
        self.extend_len = 0
        self.multi_label = True
        BaseDataset.__init__(self, opt)
        self.brats_file = h5py.File(os.path.join(opt.dataroot, h5_name), 'r')
        train_db = self.brats_file['train']
        self.dcm, self.label, self.seg = self.build_pairs(train_db)

        # self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        # self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def build_pairs(self, dataset):
        keys = dataset.keys()
        # keys = list(keys)[:2]
        dcm_arr = []
        label_arr = []
        seg_arr = []
        for key in keys:

            print(f"build key:{key}")
            sys.stdout.flush()
            dcm = dataset[f"{key}/t2"][()]
            label = dataset[f"{key}/seg"][()]

            for i in range(label.shape[2]):
                if len(np.unique(label[:, :, i])) > 1:
                    slice_dcm = dcm[:, :, i]
                    # slice_dcm = slice_dcm * ((pow(2, 16) - 1) / slice_dcm.max())
                    # slice_dcm = slice_dcm.astype("uint16")
                    slice_dcm = slice_dcm * ((pow(2, 8) - 1) / (slice_dcm.max() + 1e-8))
                    slice_dcm = slice_dcm.astype("uint8")
                    slice_label = label[:, :, i]

                    seg_label = np.copy(slice_label).astype("uint8")
                    # Need to be verify if label only 0,2
                    seg_label = seg_label * (255 / (seg_label.max() + 1e-8))
                    # seg_label = seg_label * (255 / 4)
                    if not self.multi_label:
                        seg_label[seg_label > 0] = 1

                    if (np.count_nonzero(seg_label) < 10):
                        continue

                    if self.real_tumor:
                        # slice_label[slice_label > 0] = -1
                        # slice_label = slice_label+1
                        slice_label[slice_label > 0] = 1
                        slice_label = slice_label*slice_dcm

                        slice_label = self.merge_skull(slice_dcm, slice_label, default_skull_value=255)
                    else:
                        slice_label = self.merge_skull(slice_dcm, slice_label)

                    dcm_arr.append(slice_dcm)
                    label_arr.append(slice_label)
                    seg_arr.append(seg_label)

            # start, end = _range_idx(label, self.extend_len)
            # for i in range(start, end+1):
            #     slice_dcm = dcm[:, :, i]
            #     # slice_dcm = slice_dcm * ((pow(2, 16) - 1) / slice_dcm.max())
            #     # slice_dcm = slice_dcm.astype("uint16")
            #     slice_dcm = slice_dcm * ((pow(2, 8) - 1) / slice_dcm.max())
            #     slice_dcm = slice_dcm.astype("uint8")
            #     slice_label = label[:, :, i]
            #
            #     seg_label = np.copy(slice_label).astype("uint8")
            #     seg_label = seg_label * (255 / (seg_label.max()+1e-8))
            #     if not self.multi_label:
            #         seg_label[seg_label > 0] = 255
            #     # if (np.count_nonzero(seg_label) < 10 and self.is_test):
            #     #     continue
            #
            #     slice_label = self.merge_skull(slice_dcm, slice_label)
            #
            #     dcm_arr.append(slice_dcm)
            #     label_arr.append(slice_label)
            #     seg_arr.append(seg_label)

        if self.is_test:
            factor = 4
            add_dcm_arr = []
            add_label_arr = []
            add_seg_arr = []
            if factor >= 1:
                for i in range(len(dcm_arr)):
                    dcm = dcm_arr[i]

                    times = 1

                    skull_mask = np.zeros_like(dcm)
                    skull_mask[dcm > 0] = 1
                    while times <= factor:
                        random_id = random.randrange(0, len(dcm_arr) - 1, 10)
                        seg = seg_arr[random_id]


                        # seg[seg > 0] = 1
                        seg = self.seg_in_skull(seg, skull_mask)
                        seg = seg.astype("uint8")

                        seg_mask = np.copy(seg)
                        seg_mask[seg_mask>0]=1
                        if np.sum(seg_mask) < 10:
                            continue
                        label = np.copy(seg)
                        label = label * 4 / 255

                        label = self.merge_skull(skull_mask, label)
                        # label[label > 0] = 255


                        add_dcm_arr.append(dcm)
                        add_label_arr.append(label)
                        add_seg_arr.append(seg)
                        times += 1
                    print(f'append syn label:{i}')
                print(f"##debug: orig_dcm last length:{len(dcm_arr)}")
                dcm_arr = dcm_arr + add_dcm_arr
                label_arr = label_arr + add_label_arr
                seg_arr = seg_arr + add_seg_arr
                print(f"##debug: updated_dcm last length:{len(dcm_arr)}")

        return dcm_arr, label_arr, seg_arr

    def seg_in_skull(self, seg, mask):
        # ndimage.binary_fill_holes(skull_mask)
        seg = mask * seg
        return seg

    def merge_skull(self, skull_mask, slice_label, default_skull_value=5):
        # Add skull structure into label
        skull_mask = ndimage.binary_fill_holes(skull_mask)
        skull_mask = cv2.Laplacian(skull_mask.astype("uint8"), cv2.CV_8U)
        skull_mask[skull_mask > 0] = default_skull_value
        slice_label = slice_label + skull_mask
        slice_label = slice_label * (255 / (slice_label.max() + 1e-8))
        slice_label = slice_label.astype("uint8")
        # slice_label[slice_label > 0] = 255

        return slice_label

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        A = self.label[index]
        B = self.dcm[index]
        seg = self.seg[index]
        A = Image.fromarray(A).convert('RGB')
        B = Image.fromarray(B).convert('RGB')
        seg = Image.fromarray(seg).convert('RGB')

        # read a image given a random integer index
        # AB_path = self.AB_paths[index]
        # AB = Image.open(AB_path).convert('RGB')
        # # split AB image into A and B
        # w, h = AB.size
        # w2 = int(w / 2)
        # A = AB.crop((0, 0, w2, h))
        # B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1), method=Image.NEAREST)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        seg_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1), method=Image.NEAREST)

        A = A_transform(A)
        B = B_transform(B)
        seg = seg_transform(seg)
        # seg[seg < 0] = 0

        return {'A': A, 'B': B, 'A_paths': str(index), 'B_paths': str(index), 'Seg': seg[:1, :, :]}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.dcm)
