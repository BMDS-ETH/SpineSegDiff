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

import logging
from pathlib import Path
import numpy as np
import shutil

from src.dataset import create_dataset_json
from src.utils import get_files, copy_folder, get_subdirectories, load_yaml, save_json

def create_directories(dataset_name: str, target_path: str, Path) -> [Path, Path]:
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
    return save_img_path, save_lbl_path

def copy_data_collection(dataset_path: Path, target_path: Path, folder:str= '', **kwargs)->None:
    """ Copy files from a  data collection directory to another based on a subdolfer name

    Params:
    -------
        data_path: Path
            path to the directory containing the files to be copied
        target_path: Path
            path to the directory where the files will be copied
        folder: str
            name of the folder to be copied
    """

    data_collections = [x for x in dataset_path.rglob(f'*{folder}') if x.is_dir()]
    for collection in data_collections:
        copy_folder(collection,target_path/collection.relative_to(dataset_path), **kwargs)


def structure_patients(channels:dict, ext:str, image_regex:str, labels_regex:str, modality_level:int,
                       patient_dirs_train:list, save_img_path: Path,
                           save_lbl_path:  Path, metadata = None, **kwargs):
    """ Helper function to structure the dataset
    Parameters:
    -----------
    channels: dict
        dictionary containing the name of the channels and the corresponding suffix number
    extension: str
        extension of the files to be copied
    image_regex: str
        pattern to identify the images
    labels_regex: str
        pattern to identify the labels
    modality_level: int
        level of the modality in the path
    patient_dirs_train: list
        list of the paths to the patient folders
    target_path: Path
        path to the folder where the images will be copied
    target_label_path: Path
        path to the folder where the labels will be copied
    metadata: dict
        dictionary containing the metadata of the dataset

    """

    for p in patient_dirs_train:
        patient_id = str(p).split("/")[-1]
        data_fpaths = get_files(p, pattern=image_regex, ext=ext)
        label_fpaths = get_files(p, pattern=labels_regex, ext=ext)
        for dp, lp in zip(data_fpaths, label_fpaths):
            # Get the modality of the volume_data
            mod = str(dp).rstrip(''.join(dp.suffixes)).split("/")[modality_level]
            # Encode the volume_data name with the modality as a suffix number
            m = str(channels[mod]).zfill(4)
            # Get the original volume_data name
            image_name = str(dp.stem).rstrip(''.join(dp.suffixes))
            shutil.copy2(dp, str(save_img_path / f"{patient_id}_{image_name}_{m}.{ext}"))
            shutil.copy2(lp, str(save_lbl_path / f"{patient_id}_{lp.name}"))

    return label_fpaths


def structure_patient_data(channels, ext, image_regex, labels_regex,  modality_level, p, save_img_path, save_lbl_path):
    """
    Helper function to structure the image_data of a patient in a common folder
    Parameters:
    -----------
    channels: dict
        dictionary containing the name of the channels and the corresponding suffix number
    extension: str
        extension of the files to be copied
    image_regex: str
        pattern to identify the images
    labels_regex: str
        pattern to identify the labels
    modality_level: int
        level of the modality in the path
    p: Path
        path to the patient folder
    save_img_path: Path
        path to the folder where the images will be copied
    target_label_path: Path
        path to the folder where the labels will be copied
    """

    patient_id = str(p).split("/")[-1]
    ext = ext if ext.startswith('.') else f'.{ext}'
    data_fpaths = get_files(p, pattern=image_regex, ext=ext)
    label_fpaths = get_files(p, pattern=labels_regex, ext=ext)

    for dp in data_fpaths:
        # Get the modality of the volume_data from the Path
        mod = str(dp).rstrip(''.join(dp.suffixes)).split("/")[modality_level]

        if mod not in channels.keys():
            logging.warning(f"The modality {mod} was excluded as it is not saved channels dictionary")
            continue

        # Encode the volume_data name with the modality as a suffix number
        m = str(channels[mod]).zfill(4)

        # Get the original volume_data name
        image_name = str(dp.stem).rstrip(''.join(dp.suffixes))

        # Copy the volume_data to the corresponding folder
        shutil.copy2(dp, str(save_img_path / f"{patient_id}_{image_name}_{m}{ext}"))

    for lp in label_fpaths:
        shutil.copy2(lp, str(save_lbl_path / f"{patient_id}_{lp.name}"))
    return label_fpaths


def create_dataset_dicomdir(source_path: [str, Path], dataset_dir: str, target_path: [str, Path], ext:str=".nii.gz", data_structure:str ="ID/imagesTr", **kwargs) :
    """ Runs image_data processing scripts to turn raw image_data from (..data/raw) into
        cleaned image_data ready to train (saved in ..data/processed).
        Params:
        -------
        source_path: Path
            path to the directory containing the files to be copied
        dataset_name: str
            name of the folder to be copied
        target_path: Path
            path to the directory where the files will be copied
        extension: str
            extension of the files to be copied
        data_structure: str
            structure of the image_data to be copied
    """

    all_train_files = []
    folders = data_structure.split("/")
    ID_level, image_level = 0, -1

    for f in folders:
        ID_level = folders.index(f) if "ID" in f else ID_level
        image_level = folders.index(f) if "images" in f else image_level
        modality_level = folders.index(f) if "modalities" in f else -1

    image_regex = kwargs.get('images_pattern', "*")
    labels_regex = kwargs.get('labels_pattern', "*")
    images_fpaths = get_files(Path(source_path) / dataset_dir, pattern=image_regex, ext=ext)
    # Get the modalities from the image_data structure or from the fname names if not specified in the params fname
    modalities = kwargs.get('modalities', [str(f).rstrip(''.join(f.suffixes)).split("/")[modality_level] for f in images_fpaths] )
    channels = {str(m): i for i,m in enumerate(np.unique(modalities))}
    ID_regex = (ID_level + 1) *'*/'
    patient_datadirs = get_subdirectories(Path(source_path) / dataset_dir, regex=ID_regex)

    if patient_datadirs:
        labels_map = {"background": 0}
        save_img_path, save_lbl_path = create_directories(dataset_dir, target_path)
        for p in patient_datadirs:
            label_fpaths = structure_patient_data(channels, ext, image_regex, labels_regex, modality_level, p,
                                                  save_img_path, save_lbl_path)
            labels_map.update( kwargs.get('labels_map', {str(l): i+1 for i,l in enumerate(np.unique([str(l.stem).rstrip(''.join(l.suffixes)) for l in label_fpaths]) )  } ) )

        info = load_yaml(Path(source_path) / dataset_dir / f"{kwargs['name']}.yaml")['data']
        metadata_fpaths = [m for m in info.get("metadata", []) if Path(m).exists()]

        # Copy the metadata files to the dataset folder
        for m in metadata_fpaths:
            shutil.copy2(m, str(Path(target_path) / dataset_dir / Path(m).name))

        data_fpaths = get_files(save_img_path, pattern=image_regex, ext=ext)

        # Save the dataset info in the dataset folder
        dataset_info = create_dataset_json( dataset_name = dataset_dir, training_files= data_fpaths, file_ext=ext,
                             channels = channels, labels_map = labels_map, num_files = len(data_fpaths),  **info)

        # Save the dataset info in the dataset folder
        save_json(dataset_info, "dataset.json", Path(target_path) / dataset_dir, sort=False )
