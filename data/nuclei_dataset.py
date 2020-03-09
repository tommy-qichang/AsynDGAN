import os.path

import h5py
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform


class NucleiDataset(BaseDataset):
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
            h5_name = "train_all.h5"
        else:
            if idx ==0:
                h5_name = "train_breast.h5"
            elif idx ==1:
                h5_name = "train_kidney.h5"
            elif idx ==2:
                h5_name = "train_liver.h5"
            elif idx ==3:
                h5_name = "train_prostate.h5"

        print(f"Load: {h5_name}")
        self.is_test = True
        self.real_tumor = False
        self.extend_len = 0
        self.multi_label = True
        BaseDataset.__init__(self, opt)
        self.brats_file = h5py.File(os.path.join(opt.dataroot, h5_name), 'r')

        if 'train' in self.brats_file:
            train_db = self.brats_file['train']
        else:
            train_db = self.brats_file
        self.dcm, self.label, self.labels_ternary, self.weight_maps = self.build_pairs(train_db)

        # self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        # self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def build_pairs(self, dataset):
        dcm_arr = []
        label_arr = []
        labels_ternary_arr = []
        weight_maps_arr = []

        images = dataset['images']
        labels = dataset['labels']
        labels_ternary = dataset['labels_ternary']
        weight_maps = dataset['weight_maps']

        keys = images.keys()
        # keys = list(keys)[:2]
        for key in keys:
            img = images[key][()]
            label = labels[key][()]
            label_t = labels_ternary[key][()]
            weight_m = weight_maps[key][()]

            dcm_arr.append(img)
            label_arr.append(label)
            labels_ternary_arr.append(label_t)
            weight_maps_arr.append(weight_m)

        return dcm_arr, label_arr, labels_ternary_arr, weight_maps_arr

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
        A = Image.fromarray(A).convert('RGB')
        B = Image.fromarray(B).convert('RGB')
        labels_ternary = self.labels_ternary[index]
        weight_map = self.weight_maps[index]
        labels_ternary = labels_ternary[:256, :256, :]
        weight_map = weight_map[:256, :256]

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
        transform_params['crop_pos'] = (0, 0)
        transform_params['vflip'] = False
        transform_params['hflip'] = False
        self.opt.load_size = 286
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        # seg[seg < 0] = 0

        return {'A': A, 'B': B, 'A_paths': str(index), 'B_paths': str(index),
                "label_ternary": labels_ternary,
                "weight_map": weight_map}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.dcm)
