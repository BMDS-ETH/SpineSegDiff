# -*- coding: utf-8 -*-

# Copyright (c) BLINDED
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import shutil
from pathlib import Path
import numpy as np
from typing import Union, Tuple
import re
from collections import defaultdict
from src.medimages.processing import convert_image
from src.dataset import create_dataset_json, split_index_folds
from src.utils import load_yaml, save_json, get_files, set_logger, get_environment_variable, save_yaml



def create_directories(dataset_name: str, target_path: Union[str, Path]) -> Tuple[Path, Path]:
    """ Helper function to create the directories for the dataset

    Parameters:
    ------------
    dataset_name: str
        name of the dataset
    target_path: str
        path to the directory where the dataset will be saved
    """

    save_img_path = Path(target_path) / dataset_name / "imagesTr"
    save_img_path.mkdir(parents=True, exist_ok=True)
    save_lbl_path = Path(target_path) / dataset_name / "LabelsTr"
    save_lbl_path.mkdir(parents=True, exist_ok=True)
    info_path = Path(target_path) / dataset_name / "MetadataInfo"
    info_path.mkdir(parents=True, exist_ok=True)
    return save_img_path, save_lbl_path


def args_parser():
    parser = argparse.ArgumentParser(description="Create Training Dataset")

    parser.add_argument(
        "-pf",
        "--params_file",
        type=str,
        help="path to the config filename",
        default="params.yaml")

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="config parameters",
        default="data")

    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        help="path to load the data",
        default="data/raw")

    parser.add_argument(
        "-t",
        "--target_path",
        type=str,
        help="path to save the dataset",
        default="data/processed")

    parser.add_argument(
        "-l",
        "--logs",
        type=str,
        help="level of logging",
        default="INFO")
    parser.add_argument(
        "-i",
        "--include",
        type=bool,
        help="include the SPACE images for this dataset",
        default=True)

    parser.add_argument(
        "--log_dir",
        type=str,
        help="path to the model fname",
        default="./.logs")
    return parser.parse_args()


def process_segmentation_dataset(images_fpaths: list[Path], labels_fpaths: list[Path], dst_path: [str, Path], dataset_name: str, labels_dict: dict, **kwargs) -> None:
    """ Process the dataset
        Parameters:
        -----------
        images_fpaths: list[Path]
            list of paths to the images
        labels_fpaths: list[Path]
            list of paths to the labels
        target_data_path: [str, Path]
            path to save the dataset
        dataset_name: str
            name of the dataset
        labels_dict: dict
            dictionary with the labels and the corresponding values
        kwargs: dict
            additional arguments

        Returns:
        --------
        None. The dataset is saved in the target_data_path and the dataset_dict with training samples is saved in the target_data_path/dataset.json

        Examples:
        ---------
        Images and labels are stored in the same folder with the same ID
        >>> process_segmentation_dataset(images_fpaths, labels_fpaths, dst_path, dataset_name, labels_dict)

    """
    data_samples =[]
    ids_images = defaultdict(list)
    ids_labels = defaultdict(list)

    # Find all the numeric substrings and its position in the filename
    for i, fp in enumerate(images_fpaths):
        id = re.search(r'\d+', fp.as_posix())
        ids_images[str(id.group())].append(i)

    for i, fp in enumerate(labels_fpaths):
        id = re.search(r'\d+', fp.as_posix())
        ids_labels[str(id.group())].append(i)
    n_folds = kwargs.get('split_folds', 5) if isinstance(kwargs.get('split_folds', False), int) else 1
    folds_id = split_index_folds( np.array(list(ids_images.keys())), n_folds, shuffle=True)

    
    # Find the images and labels with the same ID
    for id, index in ids_images.items():
        # Find the images and labels with the same ID
        files = np.array(images_fpaths)[index]
        labels = np.array(labels_fpaths)[ids_labels[id]]
        fold = folds_id.get(id, 0)

        if kwargs.get('merge_mask', False):
            # Make sure the labels are in the same folder
            labels = [l for l in labels if l.parent.parent == files[0].parent.parent]
            files = [f for f in files if f.parent.parent == labels[0].parent.parent]

            for f in files:
                m = [f'{m}_{c.zfill(4)}' for m, c in channels.items() if m in str(f)]
                fp = convert_image(f, dst_path, ext=extension, dataset=dataset_name, ID=id.zfill(3),
                                   modality=''.join(m), **kwargs)
                data_samples.append({'image': fp.relative_to(target_path), 'label': lp.relative_to(target_path), 'fold': fold})


        elif len(files) == len(labels):
            # Assuming that the images and labels are in the same order and filtered by the same modality
            for f, l in zip(files, labels):

                # Find the modality in the filename and add the channel number for nn-Unet compatibility
                m = [f'{m}_{c.zfill(4)}' for m, c in channels.items() if m in str(f) and str(l)]
                fp = convert_image(f, dst_path, ext=extension, dataset=dataset_name, ID=id.zfill(3),
                                   modality=''.join(m), **kwargs)
                lp = convert_image(l, target_label_path, ext=extension, dataset=dataset_name, ID=id.zfill(3),
                                   modality=''.join(m), **kwargs)
                data_samples.append({'image': fp.relative_to(target_path), 'label': lp.relative_to(target_path), 'fold': fold})
        else:

            lp = convert_image(labels, target_label_path, ext=extension, dataset=dataset_name,
                                ID=id.zfill(3), **kwargs)
            for f in files:
                m = [f'{m}_{c.zfill(4)}' for m, c in channels.items() if m in str(f)]

                fp = convert_image(f, dst_path, ext=extension, dataset=dataset_name, ID=id.zfill(3),
                                   modality=''.join(m), **kwargs)
                data_samples.append({'image': fp.relative_to(target_path), 'label': lp.relative_to(target_path) , 'fold': fold})

            logger.warning(f'ID {id} has {len(files)} images and {len(labels)} labels')

    # Save the dataset dataset_info in the dataset folder
    dataset_info = create_dataset_json(dataset_name=dataset_name, description=info.pop('info'),
                                       training_files=data_samples, file_ext=extension,
                                       channels=channels, labels_map=labels_dict, num_files=len(data_samples), **info)
    logger.info(f'Created dataset {dataset_name} with {len(data_samples)} samples')

    save_json(dataset_info, target_path / f"{dataset_name}" / "dataset.json")


if __name__ == '__main__':

    # Set the project target_dir, useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # Loads the arguments
    args = args_parser()
    kwargs = load_yaml(Path(project_dir, args.params_file))[args.config]
    # Load the image_data directory value both, from the environment variable or args if missing
    data_path = get_environment_variable('DATA_DIR', get_environment_variable('DATA_DIRECTORY', default= args.data_path)) 
    print(data_path)

    # Set the logger
    logger = set_logger(name='create_seg_dataset', log_level=args.logs, save_path=args.log_dir)

    #add the logger to the kwargs dictionary for usage later in the script
    kwargs["logger"] = logger


    extension = kwargs.get('extension', 'nii.gz')
    dataset_name = kwargs.get('name', 'dataset')
    # Load the image_data info from the yaml file

    info = load_yaml(Path("config/data")/f"{dataset_name}.yaml")[args.config]

    logger.info(f'Creating dataset {dataset_name} from {data_path}')
    # Create the target directory
    target_path = Path(project_dir, args.target_path )
    target_data_path, target_label_path = create_directories(dataset_name, target_path)

    # Get the pattern to find the images and labels
    image_regex  = info.pop('images_pattern', "*")
    labels_regex = info.pop('labels_pattern', "*")
    include = kwargs.get("include", False)
    #if there is a T2_Space image / Label pair, change the name of the T2_Space image to T2
    if include == True:
        image_regex = "/images/*"
        labels_regex = "/masks/*"
        logger.info(f'Including the SPACE images for this dataset, original raw data will be moved to a backup folder: {Path(data_path).parent}/backup')

        backup_path = os.path.join(Path(data_path).parent , 'backup') #copy raw data to a folder
        if not os.path.exists(backup_path):
            os.makedirs(backup_path)
        shutil.copytree(data_path,backup_path, dirs_exist_ok=True)

        images_fpaths = get_files(Path(data_path), pattern=image_regex, ext=info.pop("extension", ".*"))
        labels_fpaths = get_files(Path(data_path), pattern=labels_regex)

        image_f_included = []
        label_f_included = []
        logger.info(f't2 images get stored as t2_interpolated and t2_space images get stored as t2')

        for image, label in zip(images_fpaths, labels_fpaths):
            if image.stem.endswith('SPACE') and label.stem.endswith('SPACE'):
                logger.info(f'Processing image {image.stem} and label {label.stem} where we switch the T2_Space image to T2 naming careful this altered the raw data!')
                #get the number of the image

                num = re.search(r'\d+', image.stem).group()

                for img, lab in zip(images_fpaths, labels_fpaths):
                    if re.search(r'\d+', img.stem).group() == num and img.stem.endswith('t2'): #renaming the t2 images as interpolated_from_space if a space image is present for it.
                        img = img.rename(os.path.join(img.parent / f'{img.stem.replace("t2","interpolated_from_Space")}{img.suffix}'))
                        lab = lab.rename(os.path.join(lab.parent / f'{lab.stem.replace("t2","interpolated_from_Space")}{lab.suffix}'))
                        break


                image = image.rename(image.parent / f'{image.stem.replace("t2_SPACE", "t2")}{image.suffix}')
                label = label.rename(label.parent / f'{label.stem.replace("t2_SPACE", "t2")}{label.suffix}')



            image_f_included.append(image)
            label_f_included.append(label)

        image_f_included = [x for x in image_f_included if not x.stem.endswith("interpolated_from_Space")]
        label_f_included = [x for x in label_f_included if not x.stem.endswith("interpolated_from_Space")]

        images_fpaths = sorted(set(image_f_included))
        labels_fpaths = sorted(set(label_f_included))




    else:
        logger.info(f'Not including the SPACE images for this dataset but double check if the T2 images are interpolated from the SPACE images this means that the T2 Images are the SPACE images')

        images_fpaths = get_files(Path(data_path), pattern=image_regex, ext=info.pop("extension", ".*"))
        labels_fpaths = get_files(Path(data_path), pattern=labels_regex)


    #if there is a T2_Space image / Label pair, change the name of the T2_Space image to T2






    # Get the modalities from the image_data structure or from the fname names if not specified in the params fname
    modalities = info.pop('modalities', np.unique([fpath.stem.split('_')[-1] for fpath in images_fpaths]))
    channels = {str(m): str(i) for i, m in enumerate(np.unique(modalities))}

    # Get and eliminate the labels from the image_data structure or from the fname names if not specified in the params fname
    labels_dict = info.pop('labels', {str(l): i+1 for i,l in enumerate(np.unique([str(l.stem).rstrip(''.join(l.suffixes)) for l in labels_fpaths]) )  })

    # Get the metadata dataset information files from the yaml file or from the dataset folder
    metadata_fpaths = [m for m in info.pop("metadata", []) if Path(m).exists()] + [
        Path("config/data") / f"{dataset_name}.yaml"]
    # Copy the metadata files to the dataset folder


    for m in metadata_fpaths:
        shutil.copy2(m, str(Path(target_path) / dataset_name / 'MetadataInfo' / Path(m).name))

    # Process the dataset
    process_segmentation_dataset(images_fpaths, labels_fpaths, target_data_path, dataset_name, labels_dict, **kwargs)

    #create a .yaml file with the dataset information in the config/data folder
    #lets create the dictionary with the dataset information
    dataset_dict = {"name": str(dataset_name),
                    "source_path": str(data_path),
                    "target_path": str(Path(target_path) / dataset_name)}
    #save the dictionary as .yaml file
    save_yaml(dataset_dict, Path(project_dir, "config/data") / f"pipeline_summary.yaml")
    print(f"Created a .yaml file with the dataset information in the config/data folder")
    


