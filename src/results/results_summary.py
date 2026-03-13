import argparse
import pandas as pd
import logging
from pathlib import Path
import re
import matplotlib.pyplot as plt

from src.results.statistical_summary import segmentation_pathology_analysis

plt.rcParams['text.usetex'] = True

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r","--result_dir", type=str,default="models/DiffUnet2D/SPIDER_T2w/")
    parser.add_argument("-l","--log_dir", type=str,default="./.logs")
    parser.add_argument("-lvl","--log_level", type=str,default="INFO")

    
    args = parser.parse_args()
    return args


# Function to extract ID from image filename
def extract_id(image_name):
    match = re.search(r'SPIDER_(\d+)_', str(image_name))
    if match:
        return int(match.group(1))
    return None

def extract_contrast(image_name):
    """ Find the contrast type in the image name """
    match = re.search(r'_(t1|t2)_', str(image_name))
    if match:
        contrast = str(match.group(1)).upper()+"w"
        return contrast
    return None


if __name__ == "__main__":
    args = arg_parser()

    ids = [1, 50, 58, 130, 133, 152, 160, 186, 251]
    result_dir = args.result_dir
    all_folds_results = [f for f in sorted(Path(result_dir).rglob("*/dice_semantic.csv"))]
    all_dice_df = pd.DataFrame([])
    logging.info(all_folds_results)
    for index, filepath in enumerate(all_folds_results):
        #read in the dice_semantic.csv.file
        # dice_file = os.path.join(result_dir, filepath, "dice_semantic.csv")
        dice_df = pd.read_csv(filepath)

        if index == 0:
            all_dice_df = dice_df
        else:
            #concate the dice_df to the all_dice row wise without the header column
            all_dice_df = pd.concat([all_dice_df, dice_df],axis=0,ignore_index=True)
            #check if now duplicates are in the all_dice_df
            duplicates = all_dice_df[all_dice_df.duplicated()]

            if len(duplicates) > 0:
                logging.info(f'Found {len(duplicates)} duplicates in fold {index}')
                logging.info(duplicates)
                logging.info("Double check if you splits patientwise and have used the same for nnUNet otherwise"
                             "\notherwise one possiblity is that you've ran inference twice for the same fold")
                all_dice_df = all_dice_df.drop_duplicates()

    # Exclude the rows which id is one of the ids in the image filename SPIDER_ID_XXX
    # ids = [1, 50, 58, 130, 133, 152, 160, 186, 251]
    # how to thech if the ids are in the image filename part of the dataframe
    if 'epoch' in all_dice_df.columns:
        all_dice_df = all_dice_df.drop(columns=['epoch'])

    if 'Vertebrae' in all_dice_df.columns:
        #rename the column to VB
        all_dice_df.rename(columns={'Vertebrae':'VB'}, inplace=True)


    if ' SC' in all_dice_df.columns:
    #if the columns have spaces, remove them
        all_dice_df.rename(columns={' VB':'VB'}, inplace=True)
        all_dice_df.rename(columns={' SC':'SC'}, inplace=True)
        all_dice_df.rename(columns={' IVD':'IVD'}, inplace=True)

    #Compute the mean along the row for Sc, vertebral bodies and disc
    all_dice_df['DICE'] = all_dice_df[["SC","VB", "IVD" ]].mean(axis=1)
    #all_dice_df['DICE'] = all_dice_df.iloc[:,1:].mean(axis=1)

    # Apply the function to create a new column with the extracted IDs
    all_dice_df['pid'] = all_dice_df['image'].apply(extract_id)
    all_dice_df['mri'] = all_dice_df['image'].apply(extract_contrast)

    # Filter the DataFrame to exclude rows with IDs in the ids_to_exclude list
    filtered_df = all_dice_df[~all_dice_df['pid'].isin(ids)]

    all_dice_df.to_csv(Path(result_dir)/"mean_dice_semantic.csv",index=False)

    mean_dice = filtered_df.iloc[:,1:-3].mean(axis=0)
    std_dice = filtered_df.iloc[:,1:-3].std(axis=0)
    print(f"Dice scores mean per class: \n{mean_dice}")
    print(f"Std Dice scores per class: \n{std_dice}")

    print(f"Overall Dice scores \n {mean_dice.mean()} +- {std_dice.std()}")

    gradings_df = pd.read_csv('data/summary_gradings.csv')
    model = result_dir.split('/')[1]
    # segmentation_pathology_analysis(filtered_df, gradings_df, save_path=result_dir,
    #                                 title=f'Statistical Analysis of Spine Pathologies on {model} Segmentation Performance')
    segmentation_pathology_analysis(filtered_df, gradings_df, save_path=result_dir,
                                    title=f'Statistical Analysis of Spine Pathologies Segmentation Performance')
