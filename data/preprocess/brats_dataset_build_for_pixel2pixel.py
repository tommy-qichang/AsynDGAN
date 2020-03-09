import os

import cv2
import h5py
import numpngw
import numpy as np
import scipy.ndimage as ndimage


def save_pairs_segmentation(data_path, save_root):
    h5_data = h5py.File(data_path, "r")
    test_data = h5_data["test"]
    train_data = h5_data["train"]
    _save_pairs(test_data, save_root, "test")
    _save_pairs(train_data, save_root, "train")


def _range_idx(label, margin):
    uniq = []
    for i in range(label.shape[2]):
        if len(np.unique(label[:, :, i])) > 1:
            uniq.append(1)
        else:
            uniq.append(0)
    min_idx = max(0, uniq.index(1) - margin)
    upper_idx = len(uniq) - uniq[::-1].index(1) - 1
    max_idx = min(len(uniq), upper_idx + margin)
    return min_idx, max_idx


def _save_pairs(dataset, save_path, data_type):
    keys = dataset.keys()
    for key in keys:
        dcm = dataset[f"{key}/t2"][()]
        label = dataset[f"{key}/seg"][()]

        start, end = _range_idx(label, 20)
        for i in range(start, end):
            slice_dcm = dcm[:, :, i]

            slice_dcm = slice_dcm * ((pow(2, 16) - 1) / slice_dcm.max())
            slice_dcm = slice_dcm.astype("uint16")
            slice_label = label[:, :, i]

            # Add skull structure into label
            skull_mask = np.zeros_like(slice_dcm)
            skull_mask[slice_dcm > 0] = 1
            ndimage.binary_fill_holes(skull_mask)
            skull_mask = cv2.Laplacian(skull_mask.astype("uint8"), cv2.CV_8U)
            skull_mask[skull_mask > 0] = 1
            slice_label = slice_label + skull_mask
            slice_label = slice_label.astype("uint8")
            slice_label[slice_label > 0] = 255

            numpngw.write_png(os.path.join(save_path, "B", data_type, f"{key}_{i}.png"), slice_dcm)
            numpngw.write_png(os.path.join(save_path, "A", data_type, f"{key}_{i}.png"), slice_label)
            print(f"Save image to: {save_path}/A/{data_type}/{key}_{i}.png...")


# save_pairs_segmentation("/share_hd1/db/BRATS/2018/BraTS18.h5", "/share_hd1/db/BRATS/brats_p2p")
