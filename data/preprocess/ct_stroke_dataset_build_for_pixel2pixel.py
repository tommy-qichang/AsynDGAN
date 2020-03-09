import os

import h5py
import imageio
import numpngw
import numpy as np


def save_pairs_segmentation(data_path, save_root):
    ct_stroke_data = h5py.File(data_path, "r")
    test_data = ct_stroke_data["test"]
    train_data = ct_stroke_data["train"]
    _save_pairs(test_data, os.path.join(save_root, "test"))
    _save_pairs(train_data, os.path.join(save_root, "train"))


def _save_pairs(dataset, save_path):
    keys = dataset.keys()
    for key in keys:
        dcm = dataset[f"{key}/data"][()]
        label = dataset[f"{key}/label"][()]
        for i in range(dcm.shape[0]):
            slice_dcm = dcm[i]
            slice_label = label[i]
            if len(np.unique(slice_label)) > 1:
                wwwl = [40-100+1024,40+100+1024]
                slice_dcm[slice_dcm < wwwl[0]] = wwwl[0]
                slice_dcm[slice_dcm > wwwl[1]] = wwwl[1]
                slice_dcm = ((pow(2, 8) - 1) / (slice_dcm.max()-slice_dcm.min())) * (slice_dcm-slice_dcm.min())
                slice_dcm = slice_dcm.astype("uint8")
                slice_label = slice_label.astype("uint8")
                slice_label[slice_label > 0] = 255
                numpngw.write_png(os.path.join(save_path, "B", f"{key}_{i}.png"), slice_dcm)
                numpngw.write_png(os.path.join(save_path, "A", f"{key}_{i}.png"), slice_label)
                print(f"Save image to: {save_path}/A/{key}_{i}.png...")


save_pairs_segmentation("/share_hd1/db/StrokeCT.h5", "/share_hd1/db/stroke_ct")
