"""
Build hdf5 files based on MURA datasets.

"""

import os
import sys

import h5py
import nibabel as nib


def readlist(dcm_path, save_path, save_name, test_case_number):
    dcm_folders = sorted(os.listdir(dcm_path))
    result_file = h5py.File(os.path.join(save_path, save_name), "w")

    idx = 0
    for i in dcm_folders:
        if idx < test_case_number:
            type = "test"
        else:
            type = "train"
        if os.path.isdir(os.path.join(dcm_path, i)):
            sys.stdout.flush()
            flair = nib.load(os.path.join(dcm_path, i, f"{i}_flair.nii.gz")).get_fdata()
            seg = nib.load(os.path.join(dcm_path, i, f"{i}_seg.nii.gz")).get_fdata()
            t1 = nib.load(os.path.join(dcm_path, i, f"{i}_t1.nii.gz")).get_fdata()
            t1ce = nib.load(os.path.join(dcm_path, i, f"{i}_t1ce.nii.gz")).get_fdata()
            t2 = nib.load(os.path.join(dcm_path, i, f"{i}_t2.nii.gz")).get_fdata()

            result_file.create_dataset(f"{type}/{idx}/flair", data=flair)
            db = result_file.create_dataset(f"{type}/{idx}/seg", data=seg)
            result_file.create_dataset(f"{type}/{idx}/t1", data=t1)
            result_file.create_dataset(f"{type}/{idx}/t1ce", data=t1ce)
            result_file.create_dataset(f"{type}/{idx}/t2", data=t2)

            db.attrs['id'] = i

            print(f"***Finish create one database:{i},***")
            idx += 1

    result_file.close()

readlist("/share_hd1/db/BRATS/2018/HGG","/share_hd1/db/BRATS/2018","BraTS18.h5",40)
# readlist("/Users/qichang/PycharmProjects/gar_v0.1/data/BRATS/MICCAI_BraTS_2018_Data_Training/HGG",
#          "/Users/qichang/PycharmProjects/gar_v0.1/data/BRATS", "brats18.h5", 40)
