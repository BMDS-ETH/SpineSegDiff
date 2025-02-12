import argparse

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import re
import matplotlib.pyplot as plt
import seaborn as sns
from src.results.statistical_summary import segmentation_pathology_analysis

plt.rcParams['text.usetex'] = True

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r","--result_dir", type=str,default="results/LumbarSpineSegDiff-postprocess/")
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
    summary_results_df =[]
    summary_results_str =[]

    for model_dir in Path(result_dir).glob("*/"):
        all_folds_results = [f for f in sorted(Path(model_dir).rglob("*outputs/dice_semantic.csv"))]
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

        all_dice_df.to_csv(Path(model_dir)/"mean_dice_semantic.csv",index=False)

        mean_dice = filtered_df.iloc[:,1:-3].mean(axis=0)
        std_dice = filtered_df.iloc[:,1:-3].std(axis=0)

        r_str = [f"{model_dir.name}", f"{mean_dice['SC']: .3f} \pm {std_dice['SC']: .3f}",
                 f"{mean_dice['VB']: .3f} \pm {std_dice['VB']: .3f}",
                 f"{mean_dice['IVD']: .3f} \pm {std_dice['IVD']: .3f}",
                 f"{mean_dice.mean(): .3} \pm {std_dice.std(): .3f}"]

        summary_results_str.append(r_str)
        summary_results_df.append([model_dir.name, mean_dice['SC'], mean_dice['VB'], mean_dice['IVD'], mean_dice.mean()])

    summary_results_df = pd.DataFrame(summary_results_df, columns=["Model", "SC", "VB", "IVD", "mDICE"])

    # save dataframe as txt
    summary_results_str = pd.DataFrame(summary_results_str, columns=["Model", "SC", "VB", "IVD", "mDICE"])
    # sort by model name
    summary_results_str = summary_results_str.sort_values(by=['Model'])

    summary_results_str.to_latex(Path(result_dir) / "summary_results.txt", index=False)


    # Save the results to a csv file

    #Separate the model name from the results
    S = summary_results_df['Model'].str.split("-", n = 1, expand = True)
    summary_results_df['S'] = S[0]
    summary_results_df['Ts'] = S[1]

    summary_results_df.to_csv(Path(result_dir)/"summary_results.csv",index=False)
    print(summary_results_str)
    # Plot the results grouped by timesteps
    #plot the results grouped by samples with seaborn


    # Sort the results by the model name
    summary_results_df = summary_results_df.sort_values(by=['Ts'])

    sns.barplot(data=summary_results_df, x='Ts', y='mDICE', hue='S', palette='Greys')
    np.arange(5,45,5)
    plt.xticks(ticks=range(0, len(summary_results_df['Ts'].unique())), labels=np.arange(5,45,5))
    plt.legend(bbox_to_anchor=(1.02, 1),
               loc='upper left',
               borderaxespad=0.)
    plt.title('Mean Dice Scores by Sample and Timestep')
    plt.ylabel('Mean Dice Score')
    plt.xlabel('Inference Timestep (Ts)')
    plt.savefig(Path(result_dir) / "summary_results_timestep.pdf",
                bbox_inches='tight',
                dpi=300)