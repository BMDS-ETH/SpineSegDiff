import sys
new_path = r'C:\Users\thoma\thesis\lumbar-spine'
sys.path.append(new_path)

import os
import numpy as np
import nibabel as nib
import json
import argparse
from pathlib import Path
import re
import yaml
import shutil
from typing import Union
from src.utils import save_json,load_yaml,set_logger, save_yaml
from src.utils.utils import get_all_paths, create_dataset_json, create_splits_json, \
    relabel_and_save, create_yaml, create_splits_from_overview, make_dirs_and_augment, get_labelmap, filter_t1_and_t2
from sklearn.model_selection import KFold
import itertools

def args_parser():
    parser = argparse.ArgumentParser(description="preprocessing_data")

    parser.add_argument("-pf","--params_file",type=str,help="path to the config filename",default="params.yaml")
    parser.add_argument("-c","--config",type=str,help="config parameters",default="prep_nn_unet")
    parser.add_argument("-j","--json_path",type=str,help="path to load the json file",default="data/processed/SPIDER")
    parser.add_argument("-t","--type",type=str,help="type of the data",default="d3_data")
    parser.add_argument("-d","--data_path",type=str,help="path to load the data",default="data/filtered")
    parser.add_argument("-s","--save_path",type=str,help="path to save the target dataset",default="data/nnUNet_raw/")
    parser.add_argument("-m","--modality",type=str,help="modality of the data can be t1,t2,both",default="both")
    parser.add_argument("-z","--zooms",type=str,help="pixel_spacing",default="data/plots/pixelspacing.json")
    parser.add_argument("-l","--logs",type=str,help="level of logging",default="INFO")
    parser.add_argument("-sl","--number_of_slices",type=int,help="number of slices to augment",default=2)
    parser.add_argument("-r","--ratio",type =float, help="ratio of test / train ", default=0.8)
    parser.add_argument("-a","--augmentation",type =bool, help="augmentation", default=False)
    parser.add_argument("-sp","--space",type=bool,help= "whether or not space t2 images weere used",default=True)
    parser.add_argument("-o", "--overview",type=str,help="path to the overview file",default=None)
    parser.add_argument("-ss","--semantic_segmentation",type=bool,help="semantic segmentation or instance segmentation",default=False)
    parser.add_argument("-diff","--diffusion",type=bool,help="whether or not to prepare a sample set for diffusion and segmentation",default=False)
    parser.add_argument("-diff_path","--diffusion_path",type=str,help="path where the data for diffusion data gets stored",default="data/DiffCreateSeg")
    parser.add_argument("-use_test","--use_test_folder",type=bool,help="whether or not to create a test folder",default=False)
    parser.add_argument("--log_dir",type=str,help="path to the model fname",default="./.logs")

    return parser.parse_args()


def extract_one_segmentation(all_labels):
    """Extract only one segmentation mask per patient
    Parameters:
    -----------
    all_labels: list
        list of all labels
    Returns:
    --------
    all_labels: list
        list of all labels with only one segmentation mask per patient"""
    if type(all_labels) != list:
        all_labels = all_labels.tolist()

    #only keep the ones that have t2 in the name
    all_labels = [x for x in all_labels if "t2" in x]
    print(all_labels)
    
    return np.array(all_labels)





def get_all_paths_twochannel(logger, data_path, type_data, folds):
    """Get all paths of the twochannel images
    Parameters:
    -----------
    logger: logging object
        logging object
    data_path: str
        path to the data
    type_data: str
        type of the data
    folds: list
        list of all folds
    Returns:
    --------
    all_images: list
        list of all images
    all_labels: list
        list of all labels"""
    
    all_images = list()
    all_labels = list()
    for fold in folds:
        image_folder = os.path.join(data_path,type_data,fold,"imagesTr")
        label_folder = os.path.join(data_path,type_data,fold,"labelsTr")
        images = os.listdir(image_folder)
        labels = os.listdir(label_folder)
        for image in images:
            all_images.append(os.path.join(image_folder,image))
        for label in labels:
            all_labels.append(os.path.join(label_folder,label))

    all_images = sorted(all_images)
    all_labels = sorted(all_labels)

    all_images = np.array(all_images)
    all_labels = np.array(all_labels)


    logger.info(f'{len(all_images)} images and {len(all_labels)} labels present')


    all_images = get_all_twochannel(all_images)
    all_labels = get_all_twochannel(all_labels)

    #extractig only one mask per patient
    all_labels = extract_one_segmentation(all_labels)



    assert len(all_images) == len(all_labels)*2, "The number of images*2 and labels should be the same"




    return all_images,all_labels

def get_all_twochannel(all_images):
    """Get all twochannel images
    Parameters:
    -----------
    all_images: list
        list of all images
    Returns:
    --------
    twochannel_images: list
        list of all twochannel images"""
    twochannel_images = list()
        #get all id's

    t1_image = list()
    t2_image = list()

    for i in range(len(all_images)):
            #search in the all_images_list if the image contains the id and is t1 and if there exists anothjer image that also has the same id and is t

        if "t1" == all_images[i].split("/")[-1].split("_")[2]:
            t1_image.append(all_images[i])
        elif "t2"== all_images[i].split("/")[-1].split("_")[2]:
            t2_image.append(all_images[i])


    for t1 in t1_image:
        for t2 in t2_image:
            if t1.split("/")[-1].split("_")[1] == t2.split("/")[-1].split("_")[1]:
                twochannel_images.append(t1)
                twochannel_images.append(t2)
                break
    twochannel_images = list(set(twochannel_images))
    return twochannel_images

def get_ids(liste):
    """Get all id's from a list of images
    Parameters:
    -----------
    liste: list
        list of images
    Returns:
    --------
    ids: list
        list of id's
    Used as a helper function for assert_patientwise_split"""
    ids = list()
    for img in liste:
        id = img.split("_")[1]
        ids.append(id)
    return ids


def assert_patientwise_split(splits_json):
    """Asserting that we actually have a patientwise split
    Parameters:
    -----------
    splits_json: dict
        dictionary that contains the train and val split
    Returns:
    --------
    None
    Raises:
    -------
    ValueError if we have the same patient in the train and val split"""

    for split in splits_json:
        train = split["train"]
        val = split["val"]
        t_ids, v_ids = get_ids(train),get_ids(val)
    
    
        for id in list(set(t_ids)):
            for v_id in list(set(v_ids)):
                if id ==v_id:
                    raise ValueError


if __name__ == '__main__':
    

    args = args_parser()
    kwargs = load_yaml(args.params_file)[args.config]
    logger = set_logger(name = "nn_unet_prep",save_path= args.log_dir, log_level= args.logs)
    augmentation = kwargs.get("augmentation",args.augmentation)
    number_of_slices = args.number_of_slices
    #add logger to kwargs that we can pass it to other modules
    kwargs["logger"] = logger
    logger.info("Starting nn-Unet preparation")

    #open the pipelin_summary.yaml file to get the correct paths
    pipeline_yaml = load_yaml(kwargs.get("pipeline_yaml"))
    pipeline_path = kwargs.get("pipeline_yaml")
    pixelspacing = round(pipeline_yaml["pixel_spacing_value"],4)

    logger.info(f'Pixelspacing: {pixelspacing}')
    #get all variables that will be used
    data_path = pipeline_yaml["save_path"]
    type_data = kwargs.get("type",args.type)
    save_path = kwargs.get("save_path",args.save_path)
    overview_file_path = kwargs.get("overview",args.overview)
    modality = kwargs.get("modality",args.modality)
    json_folder = kwargs.get("json_path",args.json_path)

    #if we included the t2_space images
    space_t2 = kwargs.get("space",args.space)

    #if we use patientwise train test split
    patientwise = kwargs.get("patientwise",False)
    if patientwise:
        logger.info(f'we will create a patientwise train test split')

    #create a dataset with semantic segmentation or instance segmentation labels
    semantic_segmentation = kwargs.get("semantic_segmentation",args.semantic_segmentation)

    #creating 3d-volumes that only consists of the middle slice
    fake_3d = kwargs.get("fake_3d",False)

    use_test_folder = kwargs.get("use_test_folder",args.use_test_folder)
    
    nn_data_dir = save_path
    image_folder_name = "imagesTr"
    label_folder_name = "labelsTr"
    folds = os.listdir(os.path.join(data_path,type_data))
    target_ratio = args.ratio


    seg_and_diff = kwargs.get("seg_and_diff",args.diffusion)
    seg_and_diff_dir = kwargs.get("diffusion_path",args.diffusion_path)
    #getting the correct label map
    label_map, labels_nn = get_labelmap(semantic_segmentation)
    if semantic_segmentation:
        logger.info("semantic segmentation")
    else:
        logger.info("instance segmentation")

    logger.info(f'{len(folds)} folds present')
    logger.info(f"Dataset, Json file and splits.json will be created! for {modality}")



    all_images, all_labels, all_test_images, all_test_labels = get_all_paths(data_path,type_data, modality, target_ratio,overview_file_path, space_t2, use_test_folder)
    
    if seg_and_diff:
        logger.info(f'we will create a dataset for diffusion and segmentation')
        all_images, all_test_images, all_labels, all_test_labels = filter_t1_and_t2(all_images, all_test_images, all_labels, all_test_labels)
        logger.info(f'we have {len(all_images)} images and {len(all_labels)} labels and test images {len(all_test_images)} and test labels {len(all_test_labels)}')



        #check that we filter out only the images and labels where we have a T1 and T2 image present



    logger.info(f'we have {len(all_images)} images and {len(all_labels)} labels and test images {len(all_test_images)} and test labels {len(all_test_labels)}')
    total_digits = 3 #initial number of digits 

    dataset_id, dataset_dir = create_yaml(modality,nn_data_dir, type_data, save_path, logger)


    logger.info(f'Data will be stored here: {dataset_dir}')

    if augmentation == True: #check if we want to augment our data
        logger.info(f'we will augment the data with {number_of_slices * 2} images per original image')
        if modality == "t1" or modality == "t2" or modality == "both":
            #get the new images and labels id's.. starts at the last id of the original images since we will augment our data.

            start_id = len(all_images) + len(all_test_images)
            logger.info(f"Augmentation starts at ID:{start_id}")
            #make the directories for the augmented images and labels
            #save the images and labels in the correct folder augmented
            all_images, all_labels, total_digits = make_dirs_and_augment(modality, number_of_slices,folds, all_images, all_labels,type_data)


    dataset_json = create_dataset_json(all_images,all_labels, all_test_images, all_test_labels, modality,labels_nn)

    if modality == "t1" or modality == "t2":
        if overview_file_path:
            logger.info(f'we have an overview file: {overview_file_path} fold will be created according to it')
            splits_json, splits,name_dict = create_splits_from_overview(all_images,all_labels,all_test_images,all_test_images,modality,folds,total_digits,overview_file_path)
            all_images = np.concatenate((all_images, all_test_images), axis=0) #concatenate the images and labels of the test and train set 
            all_labels = np.concatenate((all_labels, all_test_labels), axis=0)
            
            image_contain, label_contain = relabel_and_save(all_images,all_labels,dataset_dir,modality,label_map,type = type_data,pixelspacing = pixelspacing,name_dict = name_dict, type_data = type_data, train = True,  fake_3d = fake_3d)

        else:

            splits_json, splits,name_dict = create_splits_json(all_images,all_labels,modality,folds,total_digits)
            _,_,name_dict_test = create_splits_json(all_test_images,all_test_labels,modality,folds,total_digits)
            image_contain, label_contain = relabel_and_save(all_images,all_labels,dataset_dir,modality,label_map,type = type_data,pixelspacing = pixelspacing,name_dict = name_dict, type_data = type_data, train = True, fake_3d = fake_3d)

            image_test, label_test = relabel_and_save(all_test_images,all_test_labels,dataset_dir,modality,label_map,type = type_data,pixelspacing = pixelspacing,name_dict = name_dict_test, type_data = type_data, train = False, fake_3d = fake_3d)

    elif modality == "both":
        logger.info(f'different modality: {modality}')

        if overview_file_path:
            logger.info(f'we have an overview file: {overview_file_path} fold will be created according to it')
            splits_json, splits,name_dict = create_splits_from_overview(all_images,all_labels,all_test_images,all_test_images,modality,folds,total_digits,overview_file_path)
            all_images = np.concatenate((all_images, all_test_images), axis=0)
            all_labels = np.concatenate((all_labels, all_test_labels), axis=0)
            image_contain, label_contain = relabel_and_save(all_images,all_labels,dataset_dir,modality,label_map,type = type_data,pixelspacing = pixelspacing,name_dict = name_dict, type_data = type_data, train = True, fake_3d = fake_3d)

        else:

            if patientwise == True:
                splits_json, splits,name_dict = create_splits_json(all_images,all_labels,modality,folds,total_digits)
                _,_,name_dict_test = create_splits_json(all_test_images,all_test_labels,modality,folds,total_digits)
                
                if seg_and_diff: #save images that are being used for the diffusion and segmentation task
                    image_contain, label_contain = relabel_and_save(all_images,all_labels,seg_and_diff_dir,modality,label_map,type = type_data,pixelspacing = pixelspacing,name_dict = name_dict, type_data = type_data, train = True, fake_3d = fake_3d)
                    image_test, label_test = relabel_and_save(all_test_images,all_test_labels,dataset_dir,modality,label_map,type = type_data,pixelspacing = pixelspacing,name_dict = name_dict_test, type_data = type_data, train = False, fake_3d = fake_3d)
                
                if use_test_folder:
                    logger.info(f'Test folder gets created')
                    image_contain, label_contain = relabel_and_save(all_images,all_labels,dataset_dir,modality,label_map,type = type_data,pixelspacing = pixelspacing,name_dict = name_dict, type_data = type_data, train = True, fake_3d = fake_3d)
                    image_test, label_test = relabel_and_save(all_test_images,all_test_labels,dataset_dir,modality,label_map,type = type_data,pixelspacing = pixelspacing,name_dict = name_dict_test, type_data = type_data, train = False, fake_3d = fake_3d)
                else:

                    all_images = np.concatenate((all_images, all_test_images), axis=0)
                    all_labels = np.concatenate((all_labels, all_test_labels), axis=0)
                    splits_json, splits,name_dict = create_splits_json(all_images,all_labels,modality,folds,total_digits)
                    logger.info(name_dict)
                    image_contain, label_contain = relabel_and_save(all_images,all_labels,dataset_dir,modality,label_map,type = type_data,pixelspacing = pixelspacing,name_dict = name_dict, type_data = type_data, train = True, fake_3d = fake_3d)

            else:
                logger.info(f'Creating only a train folder')
                logger.info(f'{len(all_images)} images and {len(all_labels)} labels present')

                splits_json, splits,name_dict = create_splits_json(all_images,all_labels,modality,folds,total_digits)
                _,_,name_dict_test = create_splits_json(all_test_images,all_test_labels,modality,folds,total_digits)
                if seg_and_diff:
                    image_contain, label_contain = relabel_and_save(all_images,all_labels,seg_and_diff_dir,modality,label_map,type = type_data,pixelspacing = pixelspacing,name_dict = name_dict, type_data = type_data, train = True, fake_3d = fake_3d)
                image_contain, label_contain = relabel_and_save(all_images,all_labels,dataset_dir,modality,label_map,type = type_data,pixelspacing = pixelspacing,name_dict = name_dict, type_data = type_data, train = True, fake_3d = fake_3d)
                logger.info(f'{image_contain} images and {label_contain} labels saved')

    elif modality == "twochannel":
        logger.info(f'Creation of a two channel dataset that consists of T1 and T2 images and predicts the segmentation mask')
        folds = os.listdir(os.path.join(data_path,type_data))
        logger.info(f'{len(folds)} folds present')


        all_images, all_labels = get_all_paths_twochannel(logger, data_path, type_data, folds)
        print(len(all_images),len(all_labels))

        if not augmentation:
            pass



        all_images = np.array(sorted(all_images))
        all_labels = np.array(sorted(all_labels))

        print(all_images)
        print(len(list(set(all_images))))

        logger.info(f'{len(all_images)} twochannel images present')
        logger.info(f'{len(all_labels)} twochannel labels present')



        image_contain, label_contain = relabel_and_save(all_images,all_labels,dataset_dir,modality,label_map,type = type_data,pixelspacing = pixelspacing,name_dict = dict(), type_data = type_data, train = True, fake_3d = fake_3d)







    logger.info(f'Saving the dataset.json and splits.json in the folder {modality}')

    splits_json_save_path = os.path.join(f"data/nnUNet_preprocessed/Dataset{dataset_id:03d}_SPIDER")
    if not os.path.exists(splits_json_save_path):
        os.makedirs(splits_json_save_path)
    
    logger.info(f'{dataset_id} will be created/changed!')

    dataset_json["name"]= f"Dataset{dataset_id:03d}_SPIDER"


         
    if patientwise: #double check if the splits are patientwise
        assert_patientwise_split(splits_json)


    #saving the dataset.json and splits.json in the correct folder
    save_json(dataset_json,os.path.join(dataset_dir+"/dataset.json"))
    if use_test_folder == True or patientwise == True:
        save_json(splits_json,os.path.join(dataset_dir+"/splits_final.json"))
        logger.info(f'splits_final.json is saved in {dataset_dir} which means that the splits were created manually')
    if modality != "twochannel":
        save_json(splits_json,splits_json_save_path+"/splits_final.json")
    
    logger.info("Finished nn-Unet preparation")

    #update the pipeline_summary.yaml file and save it
    pipeline_yaml["nn_unet_path"] = save_path
    pipeline_yaml["Dataset"] = f"Dataset{dataset_id:03d}_SPIDER"
    pipeline_yaml["modality"] = modality
    pipeline_yaml["type_data"] = type_data
    pipeline_yaml["dataset_id"] = dataset_id
    pipeline_yaml["augmentation"] = augmentation
    pipeline_yaml["number_of_slices"] = number_of_slices
    pipeline_yaml["space"] = space_t2
    pipeline_yaml["patientwise"] = patientwise
    pipeline_yaml["semantic_segmentation"] = semantic_segmentation
    pipeline_yaml["labels_nn"] = labels_nn
    pipeline_yaml["Dataset_Yaml"] = f'input_{dataset_id}_{type_data}_{modality}.yaml'
    pipeline_yaml["Run for predicting Complementary Image + Segmentation"] = args.diffusion
    pipeline_yaml["use_test_folder"] = use_test_folder


    #saving the yaml file in the correct folder
    save_yaml(pipeline_yaml,kwargs.get("pipeline_yaml"))
    logger.info(f'pipeline_summary.yaml file is updated and saved in {kwargs.get("pipeline_yaml")}')