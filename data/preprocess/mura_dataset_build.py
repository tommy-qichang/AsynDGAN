"""
Build hdf5 files based on MURA datasets.

"""

import csv
import os

import h5py
import imageio
import re


def readlist(path, file):
    with open(os.path.join(path, file)) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        list = {}
        h5_file = h5py.File(os.path.join(path, 'mura.hdf5'), 'w')
        id = 0
        for row in csv_reader:
            pic_path = row[0]
            reg_arr = re.split('/XR_|/patient|/study\d_|/', pic_path)
            if len(reg_arr) > 4:
                xray_type = reg_arr[2]
                label = reg_arr[4]
                if label == "positive":
                    label = 1
                else:
                    label = 0
                img_file = imageio.imread(os.path.join(path, pic_path))
                if len(img_file.shape) == 3:
                    img_file = img_file[:, :, 0]
                ds = h5_file.create_dataset(f"{xray_type}/{label}/{id}", data=img_file)
                ds.attrs['label'] = label
                ds.attrs['path'] = pic_path
                id += 1
                print(f"Store data: {xray_type}/{label}/{id}/ shape:{img_file.shape}, label:{ds.attrs['label']} - ({row})")

        h5_file.close()


readlist("./data/", "MURA-v1.1/train_image_paths.csv")
