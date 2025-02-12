import json
import logging
import os
from pathlib import Path
import numpy as np
from monai import transforms
import numpy as np
import logging
from typing import List, Tuple

from .SPIDER import SPIDERDataset, OneHotEncodeLabels
from .SPIDERPresegmentation import SPIDERDatasePresegmentation, OneHotEncoddeLabels_presegmentation

def get_dataloader_SPIDER(data_dir, split_path, num_classes, fold=0,
                          semantic_segmentation=True, presegmentation=False,
                          train_val_test=False, presegmentation_dir="./results/nnUnet/SPIDER_T1w_T2w/outputs"):
    """
    Get the datasets for training, validation, and testing.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.
    split_path : str
        Path to the split file.
    num_classes : int
        Number of classes in the dataset.
    fold : int, optional
        Fold number (default is 0).
    semantic_segmentation : bool, optional
        Whether to use semantic segmentation or instance segmentation (default is True).
    presegmentation : bool, optional
        Whether to use the presegmented dataset (default is False).
    train_val_test : bool, optional
        Whether to use the train, val, and test split or not (default is False).
    presegmentation_dir : str, optional
        Path to the presegmentation directory (default is "./results/nnUnet/SPIDER_T1w_T2w/outputs").

    Returns
    -------
    list
        List containing the train, validation, and test datasets.
    """
    all_image_paths = [os.path.join(data_dir, "imagesTr", d) for d in os.listdir(os.path.join(data_dir, "imagesTr"))]
    all_seg_paths = [os.path.join(data_dir, "labelsTr", d) for d in os.listdir(os.path.join(data_dir, "labelsTr"))]
    assert len(all_image_paths) == len(all_seg_paths)

    with open(split_path, "r") as f:
        dataset_splits = json.load(f)

    train_files, val_files, test_files = _get_fold_split(dataset_splits, fold)

    if presegmentation:
        train_transform = transforms.Compose([OneHotEncoddeLabels_presegmentation(keys=["label", "label_pre"], num_classes=num_classes)])
        val_transform = transforms.Compose([OneHotEncoddeLabels_presegmentation(keys=["label", "label_pre"], num_classes=num_classes)])
        seg_pred_dir = presegmentation_dir if Path(presegmentation_dir).exists() else str(data_dir)
        train_dataset = SPIDERDatasePresegmentation(train_files, data_dir, transform=train_transform, seg_pred_dir=seg_pred_dir)
        val_dataset = SPIDERDatasePresegmentation(val_files, data_dir, transform=val_transform, seg_pred_dir=seg_pred_dir)
        test_dataset = SPIDERDatasePresegmentation(test_files, data_dir, transform=val_transform, seg_pred_dir=seg_pred_dir)
    else:
        train_transform = transforms.Compose([OneHotEncodeLabels(keys=["label"], num_classes=num_classes)])
        val_transform = transforms.Compose([OneHotEncodeLabels(keys=["label"], num_classes=num_classes)])
        train_dataset = SPIDERDataset(train_files, data_dir, transform=train_transform, image_dimension=2)
        val_dataset = SPIDERDataset(val_files, data_dir, transform=val_transform, image_dimension=2)
        test_dataset = SPIDERDataset(test_files, data_dir, transform=val_transform, image_dimension=2)

    return [train_dataset, val_dataset, test_dataset]


def _get_fold_split(split: dict, fold: int) -> Tuple[List[str], List[str], List[str]]:
    """
    Get the train, validation, and test file splits for a given fold.

    Parameters
    ----------
    split : dict
        Dictionary containing the dataset splits.
    fold : int
        Fold number.

    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        Lists of train, validation, and test file names.
    """

    train_files = split[fold]["train"]
    val_files = split[fold].get("val", train_files[:int(0.1 * len(train_files))])
    test_files = split[fold].get("test", train_files[int(0.2 * len(train_files)):])

    train_files = [id + "_0000.nii.gz" for id in train_files]
    val_files = [id + "_0000.nii.gz" for id in val_files]
    test_files = [id + "_0000.nii.gz" for id in test_files]
    logging.info(f"Train: {len(train_files) + len(val_files)}, Test: {len(test_files)}")

    return train_files, val_files, test_files
