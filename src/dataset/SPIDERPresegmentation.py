import json
import os
import logging
import nibabel as nib
import numpy as np
import torch
from pathlib import Path
from monai import transforms
from torch.nn.functional import one_hot
from torch.utils.data import Dataset

from tqdm import tqdm


class OneHotEncoddeLabels_presegmentation(transforms.MapTransform):
    def __init__(self, keys, num_classes):
        super().__init__(keys)
        self.num_classes = num_classes
    def __call__(self, data):
        label = data["label"]
        label_pre = data["label_pre"]

        
        if torch.any(label!=0):
            #onehote encode and ignore zeroes 
            label = one_hot(label.long(),self.num_classes).permute(2,0,1)
            label = label[1:]
        else:
            label=label[1:0]
        
        if torch.any(label_pre!=0):
            #onehote encode and ignore zeroes 
            label_pre = one_hot(label_pre.long(),self.num_classes).permute(2,0,1)
            label_pre = label_pre[1:]
        else:
            label_pre=label_pre[1:0]

        #make it channels, batch, depth, height, width
        data["label"] = label
        data["label_pre"] = label_pre
        return data
    

class SPIDERDatasePresegmentation(Dataset):
    def __init__(self,datalist, datadir, transform= None,cache = False, seg_pred_dir = None ):
        super().__init__()
        self.datalist = datalist
        self.transform = transform
        self.datadir = datadir
        self.cache = cache
        self.seg_pred_dir = seg_pred_dir
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d  = self.read_data(datalist[i])
                self.cache_data.append(d)


    def read_data(self, data_path):

        image_path = os.path.join(self.datadir, "imagesTr", data_path)
        seg_path = os.path.join(self.datadir, "labelsTr", data_path.replace("_0000.nii.gz",".nii.gz"))

        seg_pred = os.path.join(self.seg_pred_dir, data_path.replace("_0000.nii.gz",".nii.gz"))


        image_data = nib.load(image_path).get_fdata()
        image_data = np.array(image_data)
 
    
        seg_data = nib.load(seg_path).get_fdata()
        seg_data = np.array(seg_data).astype(np.int32)

        seg_pred = nib.load(seg_pred).get_fdata()
        seg_pred = np.array(seg_pred).astype(np.int32)


        #only take the middle slice of the image in the 0-th dimension
        image_data = image_data[image_data.shape[0]//2,:,:]
        seg_data = seg_data[seg_data.shape[0]//2,:,:]
        seg_pred = seg_pred[seg_pred.shape[0]//2,:,:]

        image_data = torch.as_tensor(image_data)
        image_data = image_data.unsqueeze(0)

        seg_data = torch.as_tensor(seg_data)
        seg_pred = torch.as_tensor(seg_pred)



        assert seg_data.shape == seg_pred.shape,  f'shape of seg_data {seg_data.shape} and seg_pred {seg_pred.shape} are not equal'


        return {
            "image": image_data,
            "label": seg_data,
            "label_pre": seg_pred
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
