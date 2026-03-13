from sklearn.model_selection import KFold 
import os
import json
import math
import numpy as np
import nibabel as nib
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import json
from monai import transforms
from torch.nn.functional import one_hot

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

        if torch.any(label!=0):
            #onehote encode and ignore zeroes 
            label = one_hot(label.long(),self.num_classes).permute(3,0,1,2)
            label = label[1:]
        else:
            label=label[1:0]
        #make it channels, batch, depth, height, width
        data["label"] = label
        return data


class OneHotEncode_and_Merge_only_lumbar(transforms.MapTransform):
    def __init__(self,keys,num_classes):
        super().__init__(keys)
        self.num_classes = num_classes
    def __call__(self, data):
        label = data["label"]


        #assignbackground and shift the labels from 11 - > 8 12 -> 9 13 -> 10 14 -> 11 15 -> 12 16 -> 13 17 ->
        label[(label >= 8) & (label <= 10)] = 0
        label[(label >= 11) & (label <= 19)] = label[(label >= 11) & (label <= 19)] - 3
        label[label >=14] = 0

        if torch.any(label!=0):
            #onehote encode and ignore zeroes 
            label = one_hot(label.long(),self.num_classes).permute(3,0,1,2)
            label = label[1:]
        else:
            label=label[1:0]
        #make it channels, batch, depth, height, width
        data["label"] = label
        return data




class OneHotEncoddeLabels(transforms.MapTransform):
    def __init__(self, keys, num_classes):
        super().__init__(keys)
        self.num_classes = num_classes
    def __call__(self, data):
        label = data["label"]

        
        if torch.any(label!=0):
            #onehote encode and ignore zeroes 
            label = one_hot(label.long(),self.num_classes).permute(3,0,1,2)
            label = label[1:]
        else:
            label=label[1:0]


        #make it channels, batch, depth, height, width
        data["label"] = label
        return data


class SPIDERDataset(Dataset):
    def __init__(self,datalist,datadir, transform= None,cache = False ):
        super().__init__()
        self.datalist = datalist
        self.transform = transform
        self.datadir = datadir
        self.cache = cache
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
 
    
        seg_data =nib.load(seg_path).get_fdata()
        seg_data = np.array(seg_data).astype(np.int32)
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
    
class Args:
    def __init__(self) -> None:
        self.workers=8
        self.fold=0
        self.batch_size=2
    
def get_loader_spider(data_dir,split_path,num_classes,batch_size = 1, fold = 0, semantic_segmentation = True):

    all_image_dirs = os.listdir(os.path.join(data_dir,"imagesTr"))
    all_image_paths = [os.path.join(data_dir,"imagesTr",d) for d in all_image_dirs]
    all_seg_dirs = os.listdir(os.path.join(data_dir,"labelsTr"))
    all_seg_paths = [os.path.join(data_dir,"labelsTr",d) for d in all_seg_dirs]
    assert len(all_image_paths) == len(all_seg_paths)


    #read in the dataset.json file
    with open(os.path.join(data_dir,"dataset.json"),"r") as f:
        dataset = json.load(f)
    
    label_dict = dataset["labels"]
    #get numerical list of labels
    label_list = [label for label in label_dict.keys()]

    if len(label_list) == 4:
        needs_merge = False
    
    elif len(label_list)>4 and semantic_segmentation == True:
        needs_merge = True
    else:
        needs_merge = False


    with open(split_path,"r") as f:
        split = json.load(f)
    #load the json file where the folds are specified, afterwards we can get the train, val and test splits

    if needs_merge:
        print("Merging labels for semantic segmentation")

    train_files = split[fold]["train"]
    train_files = [image + "_0000.nii.gz" for image in train_files]

    val_files = split[fold]["val"]
    val_files = [image + "_0000.nii.gz" for image in val_files]
    test_files = val_files


    print(num_classes)

    if needs_merge:
        print("Merging labels for semantic segmentation")
        train_transform= transforms.Compose([OneHotEncode_and_Merge_Labels(keys=["label"],num_classes=num_classes),
                                            transforms.RandSpatialCropd(keys=["image", "label"], roi_size=[32, 128, 128], random_size=False),
                                            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(32, 128, 128))])
        val_transform = transforms.Compose([OneHotEncode_and_Merge_Labels(keys=["label"],num_classes=num_classes)])


    else: 
        train_transform = transforms.Compose([OneHotEncoddeLabels(keys=["label"],num_classes=num_classes),
                                            transforms.RandSpatialCropd(keys=["image", "label"], roi_size=[32, 128, 128], random_size=False),
                                            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(32, 128, 128))])

                                                
        val_transform = transforms.Compose([OneHotEncoddeLabels(keys=["label"],num_classes=num_classes)
                                       ])

    train_dataset = SPIDERDataset(train_files,data_dir, transform = train_transform)
    val_dataset = SPIDERDataset(val_files, data_dir, transform= val_transform)
    test_dataset = SPIDERDataset(test_files, data_dir, transform=val_transform)

    loader = [train_dataset,val_dataset,test_dataset]

    return loader



    
