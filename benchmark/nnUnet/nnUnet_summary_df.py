import json

import numpy as np
import pandas as pd
import os
import argparse
import re
from pathlib import Path
from src.results.statistical_summary import segmentation_pathology_analysis


def extract_id(image_name):
    match = re.search(r'SPIDER_(\d+)', image_name)
    if match:
        return int(match.group(1))
    return None

def extract_contrast(image_name):
    """ Find the contrast type in the image name """
    match = re.search(r'_(t1|t2)', image_name)
    if match:
        contrast = str(match.group(1)).upper()+"w"
        return contrast
    return None


def parse_filename(filename):
    # Extract information from the filename
    parts = filename.split('_')
    pid = parts[1]
    modality = parts[2][:2]  # T1 or T2
    return pid, modality

def generate_summary(json_file_path, contrast='T1w'):
    # Load the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Initialize a list to store the extracted metrics
    summary_list = []

    # Iterate over each case in the JSON data
    for case in data['metric_per_case']:
        prediction_file = case['prediction_file']
        image_name = os.path.basename(prediction_file).replace('.nii.gz', '')
        pid = extract_id(image_name)

        modality = extract_contrast(image_name)

        # Extract metrics for SC, VB, IVD (assuming class IDs 1, 2, 3 correspond to these)
        SC_dice = case['metrics']['1']['Dice']
        VB_dice = case['metrics']['2']['Dice']
        IVD_dice = case['metrics']['3']['Dice']
        overall_dice = np.mean([SC_dice ,  VB_dice, IVD_dice])

        summary_list.append({
            'image': image_name,
            'SC': SC_dice,
            'VB': VB_dice,
            'IVD': IVD_dice,
            'DICE': overall_dice,
            'pid': pid,
            'mri': modality if modality else contrast
        })

    # Convert the list of metrics to a DataFrame
    df = pd.DataFrame(summary_list)
    return df


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r","--result_dir", type=str,default="results/nnUnet3D/SPIDER_T1wT2w/")
    parser.add_argument("-l","--log_dir", type=str,default="./.logs")
    parser.add_argument("-lvl","--log_level", type=str,default="INFO")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = arg_parser()
    contrast = extract_contrast(args.result_dir)
    json_file_path = Path(args.result_dir)/'summary.json'
    summary_df = generate_summary(json_file_path, contrast)

    # Save the DataFrame to a CSV file
    summary_df.to_csv( Path(args.result_dir)/'mean_dice_semantic.csv', index=False)


    ids = [1, 50, 58, 130, 133, 152, 160, 186, 251]
    result_dir = args.result_dir
    all_folds_results = [f for f in sorted(Path(result_dir).rglob("mean_dice_semantic.csv"))]

    for index, filepath in enumerate(all_folds_results):
        # read in the dice_semantic.csv.file
        # dice_file = os.path.join(result_dir, filepath, "dice_semantic.csv")
        dice_df = pd.read_csv(filepath)

        if index == 0:
            all_dice_df = dice_df
        else:
            # concate the dice_df to the all_dice row wise without the header column
            all_dice_df = pd.concat([all_dice_df, dice_df], axis=0, ignore_index=True)

            if len(all_dice_df[all_dice_df.duplicated()]) > 0:
                all_dice_df = all_dice_df.drop_duplicates()

        # Exclude the rows which id is one of the ids in the image filename SPIDER_ID_XXX
        # ids = [1, 50, 58, 130, 133, 152, 160, 186, 251]
        # how to thech if the ids are in the image filename part of the dataframe
        if 'epoch' in all_dice_df.columns:
            all_dice_df = all_dice_df.drop(columns=['epoch'])
        # Compute the mean allong the row for Sc, vertebral bodies and disc
        all_dice_df['DICE'] = all_dice_df[["SC", "VB", "IVD"]].mean(axis=1)
        # all_dice_df['DICE'] = all_dice_df.iloc[:,1:].mean(axis=1)

        # Apply the function to create a new column with the extracted IDs
        all_dice_df['pid'] = all_dice_df['image'].apply(extract_id)
        all_dice_df['mri'] = all_dice_df['image'].apply(extract_contrast)

        # Filter the DataFrame to exclude rows with IDs in the ids_to_exclude list
        filtered_df = all_dice_df[~all_dice_df['pid'].isin(ids)]

        all_dice_df.to_csv(Path(result_dir) / "mean_dice_semantic.csv", index=False)

    mean_dice = filtered_df.iloc[:, 1:-3].mean(axis=0)
    std_dice = filtered_df.iloc[:, 1:-3].std(axis=0)
    print(f"Dice scores mean per class: \n{mean_dice}")
    print(f"Std Dice scores per class: \n{std_dice}")

    print(f"Overall Dice scores \n {mean_dice.mean()} +- {std_dice.std()}")

    gradings_df = pd.read_csv('data/summary_gradings.csv')
    model = result_dir.split('/')[-3]
    segmentation_pathology_analysis(filtered_df, gradings_df, save_path=result_dir,
                                    title=f'Statistical Analysis of Spine Pathologies on {model} Segmentation Performance')