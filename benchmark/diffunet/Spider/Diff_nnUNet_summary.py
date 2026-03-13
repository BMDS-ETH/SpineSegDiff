import argparse
import os
import pandas as pd
import numpy as np


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--timesteps", type=int, default="100")
    parser.add_argument("-r","--result_dir", type=str,default="./results/DiffUNet")
    parser.add_argument("-")
    parser.add_argument("-p","--precondition", type=bool,default=True)
    parser.add_argument("-e","--ensemble", type=int,default=5)
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = arg_parser()

    timestep=args.timesteps
    precondition = "True" if args.precondition else "False"
    ensemble = args.ensemble


    result_dir = args.result_dir
    all_folds_results = os.listdir(result_dir)
    all_folds_results = [i for i in all_folds_results if f"timestep_{timestep}_pre_{precondition}_ensemble_{ensemble}" in i]
    print(all_folds_results)
    print(f'Found {len(all_folds_results)} results for timestep_{timestep}_pre_{precondition}_ensemble_{ensemble}')
    for index, folder in enumerate(all_folds_results):
        #read in the dice_semantic.csv.file
        dice_file = os.path.join(result_dir,folder,"dice_semantic.csv")
        dice_df = pd.read_csv(dice_file)
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



    print(all_dice_df.describe())








