from sklearn.model_selection import KFold 
import os
import json
import math
import numpy as np
import nibabel as nib
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import re
import json
from monai import transforms
import SimpleITK as sitk
from torch.nn.functional import one_hot
import multiprocessing as mp



class OneHotEncode_and_Merge_Labels(transforms.MapTransform):
    def __init__(self,keys,num_classes):
        super().__init__(keys)
        self.num_classes = num_classes
    def __call__(self, data):
        label = data["label"]


        #TODO: Merge labels (2,3,4,5,6,7,8,9,10) -> 2
        label[(label >= 2) & (label <= 10)] = 2
        # Merge labels (11, 12, 13, 14, 15, 16, 17, 18, 19) to 3
        label[(label >= 11) & (label <= 19)] = 3


        label = one_hot(label.long(),self.num_classes).permute(2,0,1)
        #make it channels, batch, depth, height, width
        data["label"] = label[1:] #exclude the background
        return data
    


class OneHotEncoddeLabels(transforms.MapTransform):
    def __init__(self, keys, num_classes):
        super().__init__(keys)
        self.num_classes = num_classes
    def __call__(self, data):
        label = data["label"]
        
        label = one_hot(label.long(),self.num_classes).permute(2,0,1)


        #make it channels, batch, depth, height, width
        data["label"] = label[1:] #exclude the background
        return data
    


class SPIDERDataset(Dataset):
    def __init__(self,datalist,datadir, transform= None, image_dimension = 3):
        super().__init__()
        self.datalist = datalist
        self.transform = transform
        self.datadir = datadir
        self.image_dimension = image_dimension

    def read_data(self, data_path, ):
        image_path = os.path.join(self.datadir, "imagesTr", data_path)
        seg_path = os.path.join(self.datadir, "labelsTr", data_path.replace("_0000.nii.gz",".nii.gz"))


        image_data = nib.load(image_path).get_fdata()
        image_data = np.array(image_data)
        seg_data =nib.load(seg_path).get_fdata()
        seg_data = np.array(seg_data).astype(np.int32)


        assert image_data.shape == seg_data.shape, "image and segmentation must have the same shape"

        if self.image_dimension == 2:
            image_data = image_data[image_data.shape[0]//2,:,:]
            seg_data = seg_data[seg_data.shape[0]//2,:,:]
    
        image_data = torch.as_tensor(image_data)
        image_data = image_data.unsqueeze(0) 
    
        seg_data = torch.as_tensor(seg_data)
        #change the label to -1 and 1 instead of 0 and 1
        #seg_data = seg_data * 2 - 1

        return {
            "image": image_data,
            "label": seg_data,
        }
    
    def __getitem__(self, index):

        image= self.read_data(self.datalist[index])

        if self.transform is not None:
            image = self.transform(image)

        seg = image["label"]
        image = image["image"]
        path = self.datalist[index]


        return image, seg, path
    
    def __len__(self):
        return len(self.datalist)
    

def get_loader_spider(data_dir,split_path, num_classes  = 4, semantic_segmentation = True, image_dimension = 2, fold = 0):
    """
    Get the datasets for training, validation and testing
    Parameters:
    ------------    
    data_dir: str
        path to the data directory
    split_path: str
        path to the json file where the folds are specified
    num_classes: int
        number of classes in the dataset
    semantic_segmentation: bool
        whether to use semantic segmentation or instance segmentation
    image_dimension: str
        whether to use 2d or 3d images
    fold: int
        fold to use for training, validation and testing
    
    Returns:
    ------------
    loader: list
        list of the datasets for training, validation and testing
    
    Examples:
    ------------
    tr_ds, val_ds, ts_ds, = get_loader_spider(data_dir,split_path, num_classes  = 4, semantic_segmentation = True, image_dimension = "3d", fold = 0)"""
    

    all_image_dirs = os.listdir(os.path.join(data_dir,"imagesTr"))
    all_image_paths = [os.path.join(data_dir,"imagesTr",d) for d in all_image_dirs]
    all_seg_dirs = os.listdir(os.path.join(data_dir,"labelsTr"))
    all_seg_paths = [os.path.join(data_dir,"labelsTr",d) for d in all_seg_dirs]
    assert len(all_image_paths) == len(all_seg_paths)

    if os.path.exists(split_path) == False:
        #create the split manually, make them always the same
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        all_image_dirs = np.array(all_image_dirs)
        all_seg_dirs = np.array(all_seg_dirs)
        split = {}
        for i, (train_index, test_index) in enumerate(kf.split(all_image_dirs)):
            train = all_image_dirs[train_index]
            test = all_image_dirs[test_index]
            split[i] = {"train": train.tolist(), "val": test.tolist()}

    else:
        with open(split_path,"r") as f:
            split = json.load(f)
    #load the json file where the folds are specified, afterwards we can get the train, val and test splits


    
    #get the train, val and test splits according to the fold that has been passed as an argument
    train_split = split[fold]["train"]
    val_split = split[fold]["val"]

    print(train_split)
    if os.path.exists(split_path) == False:
        train_files = [image for image in train_split]
        val_files = [image for image in val_split]
        test_files = [image for image in val_split]
    else:
        train_files = [image + "_0000.nii.gz" for image in train_split]
        val_files = [image + "_0000.nii.gz" for image in val_split]

        test_files = [image + "_0000.nii.gz" for image in val_split]

    
    if semantic_segmentation:
        train_transform = transforms.Compose([OneHotEncoddeLabels(keys=["label"],num_classes=num_classes)])                                     
        val_transform = transforms.Compose([OneHotEncoddeLabels(keys=["label"],num_classes=num_classes)])
    else:
        if num_classes > 4:
            train_transform = transforms.Compose([OneHotEncoddeLabels(keys=["label"],num_classes=num_classes),
                                                  transforms.CropForegroundd(keys=["image", "label"], source_key="image")])                                       
            val_transform = transforms.Compose([OneHotEncoddeLabels(keys=["label"],num_classes=num_classes),
                                                transforms.CropForegroundd(keys=["image", "label"], source_key="image")])
        else:
            raise ValueError("num_classes must be greater than 4 for instance segmentation")


    train_dataset = SPIDERDataset(train_files,data_dir, transform = train_transform, image_dimension=image_dimension)
    val_dataset = SPIDERDataset(val_files, data_dir, transform= val_transform, image_dimension=image_dimension)
    test_dataset = SPIDERDataset(test_files, data_dir, transform = None, image_dimension=image_dimension)

    loader = [train_dataset,val_dataset,test_dataset]

    return loader


