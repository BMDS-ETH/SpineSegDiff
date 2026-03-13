import argparse
import os
import pandas as pd
import numpy as np


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r","--result_dir", type=str,default="./results/both/")
    parser.add_argument("-e", "--epochs", type=int, default=1800)
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = arg_parser()
    print(args.result_dir, "result_dir")
    print(args.epochs, "epochs")




    result_dir = args.result_dir
    all_folds_results = os.listdir(result_dir)
    for index, folder in enumerate(all_folds_results):
        #read in the dice_semantic.csv.file
        dice_file = os.path.join(result_dir,folder,"dice_scores_train.csv")
        dice_df = pd.read_csv(dice_file)
        #select only where Epoch == epochs
        dice_df = dice_df[dice_df["epoch"] == args.epochs]
        print(index)
        if index == 0:
            all_dice_df = dice_df
        else:
            #concate the dice_df to the all_dice row wise without the header column
            all_dice_df = pd.concat([all_dice_df,dice_df],axis=0,ignore_index=True)
            #check if now duplicates are in the all_dice_df
            duplicates = all_dice_df[all_dice_df.duplicated()]
            if len(duplicates) > 0:
                print(f'Found {len(duplicates)} duplicates in fold {index}')
                print(duplicates)
                print("Double check if you splits patientwise and have used the same for nnUNet otherwise\notherwise one possiblity is that you've ran inference twice for the same fold")
        #delete the duplicates
                all_dice_df = all_dice_df.drop_duplicates()
        
    #get the mean and std of the dice scores for each column seperatly except the first column

# Assuming 'all_dice_df' is your DataFrame and it's already been loaded.

    # List of IDs you want to filter out
    ids = [1, 50, 58, 130, 133, 152, 160, 186, 251]

    # Generate the formatted strings for these IDs
    str_id = ['SPIDER_{:03d}'.format(i) for i in ids]

    # Filter out the rows where 'path' matches any of the strings in 'str_id'
    all_dice_df = all_dice_df[~all_dice_df['path'].str.contains('|'.join(str_id))]



    print(all_dice_df.describe())








