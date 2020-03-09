"""
Build hdf5 files based on MURA datasets.

"""

import os
import sys

import h5py
import imageio
import numpy as np
import pydicom


def readlist(dcm_path, annotation_path, save_path, test_case_number):
    dcm_folders = os.listdir(dcm_path)
    RESULT_FILE_NAME = "StrokeCT.h5"
    result_file = h5py.File(os.path.join(save_path, RESULT_FILE_NAME), "w")

    idx = 0
    for i in dcm_folders:

        if idx < test_case_number:
            type = "test"
        else:
            type = "train"

        if os.path.isdir(os.path.join(dcm_path, i)):
            dcms = sorted(os.listdir(os.path.join(dcm_path, i)))
            dcm_3d = None
            label_3d = None
            for dcm_name in dcms:
                sys.stdout.flush()
                try:
                    dcm = pydicom.dcmread(os.path.join(dcm_path, i, dcm_name))
                    dcm_array = dcm.pixel_array
                    dcm_array = np.expand_dims(dcm_array, 0)
                    if dcm_3d is None:
                        dcm_3d = dcm_array
                        first_pos = float(dcm.ImagePositionPatient[2])
                        xy_spacing = float(dcm.PixelSpacing._list[0])
                        z_spacing = float(dcm.SliceThickness)
                        intercept = float(dcm.RescaleIntercept)
                        slope = float(dcm.RescaleSlope)
                    else:
                        dcm_3d = np.concatenate((dcm_3d, dcm_array), axis=0)

                    label_name = dcm_name.replace("dcm", "png")
                    label_file_path = os.path.join(annotation_path, i, label_name)
                    if os.path.isfile(label_file_path):
                        label_file = np.expand_dims(imageio.imread(label_file_path)[:, :, 0], 0)
                        label_3d = np.concatenate((label_3d, label_file), axis=0)
                        print(f"Find Labels:{np.unique(label_file)} in file:{i}/{label_name}")
                    else:
                        if label_3d is None:
                            label_3d = np.zeros_like(dcm_3d)
                        else:
                            label_3d = np.concatenate((label_3d, np.zeros_like(dcm_array)), axis=0)

                    # print(
                    #     f"Dcm:{i}/{dcm_name}, {dcm_3d.shape}, spacing: {xy_spacing}, first_pos:{first_pos}, label:{label_3d.shape}, spacing:{xy_spacing}")
                except Exception as e:
                    print(e)
            if "ImagePositionPatient" not in dcm:
                last_pos = 10000
                print("Maybe some error?")
            else:
                last_pos = float(dcm.ImagePositionPatient[2])
            if first_pos > last_pos:
                dcm_3d = dcm_3d[::-1, :, :]
                label_3d = label_3d[::-1, :, :]
                print(f"first pos:{first_pos}, last pos:{last_pos}, reverse position.")

            result_file.create_dataset(f"{type}/{idx}/data", data=dcm_3d)
            label_db = result_file.create_dataset(f"{type}/{idx}/label", data=label_3d)
            label_db.attrs['xy_spacing'] = xy_spacing
            label_db.attrs['z_spacing'] = z_spacing
            label_db.attrs['intercept'] = intercept
            label_db.attrs['slope'] = slope
            label_db.attrs['id'] = i
            print(f"***Finish create one database:{idx}, spacing:{xy_spacing},{z_spacing}, size:{dcm_3d.shape}***")
            idx += 1


readlist("/share_hd1/db/Dicom_new3/1stUpload", "/share_hd1/db/annotations", "/share_hd1/db/", 21)
