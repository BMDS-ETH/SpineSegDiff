import os
import logging
import nibabel as nib
import numpy as np
import torch
from monai import transforms
from torch.nn.functional import one_hot
from torch.utils.data import Dataset

from tqdm import tqdm




class SPIDERDataset(Dataset):
    """
    Dataset class for the SPIDER dataset that reads in the data and the labels
    snf applies a transform to the data following the MONAI API

    Parameters:
    ------------
    datalist: list
        list of the data files
    datadir: str
        path to the data directory
    transform: callable
        transform to apply to the data
    cache: bool
        whether to cache the data
    image_dimension: int
        dimension of the image

    """
    def __init__(self, datalist, datadir, transform= None,cache = False , image_dimension = 2):
        super().__init__()
        self.datalist = datalist
        self.transform = transform
        self.datadir = datadir
        self.cache = cache
        self.image_dimension = image_dimension
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d  = self.read_data(datalist[i])
                self.cache_data.append(d)


    def read_data(self, data_path):

        image_path = os.path.join(self.datadir, "imagesTr", data_path)
        seg_path = os.path.join(self.datadir, "labelsTr", data_path.replace("_0000.nii.gz",".nii.gz"))


        image_data = nib.load(image_path).get_fdata()
        image_data = np.array(image_data)
 
    
        seg_data = nib.load(seg_path).get_fdata()
        seg_data = np.array(seg_data).astype(np.int32)

        #make them 2d
        if self.image_dimension == 2:
            #only take the middle slice of the image in the 0-th dimension
            image_data = image_data[image_data.shape[0]//2,:,:]
            seg_data = seg_data[seg_data.shape[0]//2,:,:]



        image_data = torch.as_tensor(image_data)
        image_data = image_data.unsqueeze(0)

        seg_data = torch.as_tensor(seg_data)

        return {
            "image": image_data,
            "label": seg_data,
        }
    
    
    def __getitem__(self, index):
        if self.cache:
            image = self.cache_data[index]
        else:
            image = self.read_data(self.datalist[index])


        if self.transform is not None:
            image = self.transform(image)
        return (image, self.datalist[index])
    
    def __len__(self):
        return len(self.datalist)


class OneHotEncodeLabels(transforms.MapTransform):
    def __init__(self, keys, num_classes):
        super().__init__(keys)
        self.num_classes = num_classes
        logging.info(f"{self.num_classes} classes get encoded to the dataset")

    def __call__(self, data):
        label = data["label"]

        if torch.any(label!=0):
            #onehote encode and ignore zeroes
            label = one_hot(label.long(),self.num_classes).permute(2,0,1)
            label = label[1:]
        else:
            label=label[1:0]

        #make it channels, batch, depth, height, width
        data["label"] = label

        return data