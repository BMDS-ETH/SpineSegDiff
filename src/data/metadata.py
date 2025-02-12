# Copyright (c) MA, BMDSLab ETH Zurich
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
from typing import Optional, Union, Tuple, List
import logging

def get_json_example_schema():
    """ Returns the json schema for the dataset.json fname compatible with nnU-Net

    Returns:
    --------
        dict: json schema
    """

    dataset_json = {
        "name": "",
        "description": "",
        "reference": "",
        "licence": "",
        "relase": "0.1",
        "tensorImageSize": "3D",
        "channel_names": {  # Channel names must map the index to the name of the modality, example:
            "0": "T1"
        },
        "labels": {  # nnU-Net expects consecutive params and 0 to be background
            "0": "background",
            "1": "class1",
        },
        "numTraining": 1,
        "numTest": 0,
        "training":
            [{"image": "./imagesTr/image001.nii.gz", "label": "./labelsTr/image003.nii.gz"},
             {"image": "./imagesTr/image002.nii.gz", "label": "./labelsTr/image002.nii.gz"}],
        "test": ["./imagesTs/image001.nii.gz", "./imagesTs/image002.nii.gz"]}
    return dataset_json


def create_dataset_json(dataset_name: str, training_files: List[dict], channels: dict, labels_map: dict,
                        num_files: int, file_ext: str, label_ext: Optional[str] = None, image_dim: str = "3D",
                        test_files: Optional[List[str]] = None, task: Optional[str] = "segmentation",
                        description: Optional[str] = None, regions_class_order: Optional[Tuple[int, ...]] = None,
                        overwrite_image_reader_writer: Optional[str] = None,
                        **kwargs):
    """ Creates a dataset.json fname compatible with nnU-Net

    Parameters:
    -----------
    dataset_name: str
        name of the dataset
    training_files: List[dict]
        list of dictionaries containing the paths to the images and labels
    channels: dict
        dictionary containing the channel names and the corresponding index
    labels_map: dict
        dictionary containing the labels and the corresponding index. Example:
        {
            0: 'T1',
            1: 'CT'
        }

    num_files: int
        number of training samples
    file_ext: str
        fname extension of the images and labels. This is used to find the files in the dataset folder. For the case of nnU-Net must match

    image_dim: str
        dimension of the tensor images
    test_files: List[str]
        list of paths to the test images
    task: Optional[str]
        task of the dataset
    description: Optional[str]
        brief information describing the dataset
    regions_class_order: Optional[Tuple[int, ...]]
        order of the classes in the regions.json fname. This is only required if regions are defined in the labels_map
        Example region-based training:
        {
            'background': 0,
            'whole tumor': (1, 2, 3),
            'tumor core': (2, 3),
            'enhancing tumor': 3
        }
        In this case, the regions_class_order should be (0, 1, 2, 3)
    overwrite_image_reader_writer: Optional[str]
        overwrite the default volume_data reader and writer. If you need a special IO class for your dataset you can derive it from
    BaseReaderWriter, place it into nnunet.imageio and reference it here by name
    ** kwargs: dict
        additional arguments to be added to the json fname. It can be used to add additional information about the dataset
        as long as it is compatible with json schema and does not conflict with the existing keys.

    Returns:
    --------
        dict: json schema

    """

    dataset_json = OrderedDict()  # Empty dictionary to store the dataset information in the insert order
    dataset_json['name'] = dataset_name
    dataset_json['description'] =  description if description is not None else ""
    dataset_json['task'] =  task # Segmentation by default
    # Initialize basic information of the dataset and eliminate for duplicity
    dataset_json['reference'] = kwargs.pop('doi', kwargs.get('url', ""))
    dataset_json['licence'] = kwargs.pop('licence', "CC BY-NC-SA 4.0")
    dataset_json['release'] = kwargs.pop('release', "0.0.1")

    # Check if the necessary data is provided for nnUnet
    dataset_json['channel_names'] = channels
    dataset_json['tensorImageSize'] = image_dim
    dataset_json['file_ending'] = file_ext
    dataset_json['label_extension'] = label_ext if label_ext is not None else file_ext
    dataset_json['labels'] = labels_map

    # Check if regions are defined and if so, check_online if regions_class_order is defined
    if any([isinstance(i, (tuple, list)) and len(i) > 1 for i in labels_map.values()]):
        if regions_class_order is not None:
            dataset_json['regions_class_order'] = regions_class_order
        else:
            logging.warning(
                "You have defined regions but regions_class_order is not set. This may lead to errors in nnU-Net")

    # Include the training and test files
    dataset_json['numTraining'] = num_files
    dataset_json['training'] = [ {str(k):str(v) for k,v in f.items() } for f in training_files]
    if test_files is not None:
        dataset_json['test'] =  [ {str(k):str(v) for k,v in f.items() } for f in test_files]

    if overwrite_image_reader_writer is not None:
        dataset_json['overwrite_image_reader_writer'] = overwrite_image_reader_writer

    dataset_json.update(kwargs)

    return dataset_json