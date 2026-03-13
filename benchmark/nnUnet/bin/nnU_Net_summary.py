import argparse
import os
import sys
new_path = r'C:\Users\thoma\thesis\lumbar-spine'
sys.path.append(new_path)
from src.utils import save_json,load_yaml,set_logger, save_yaml
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import yaml
import nibabel as nib
import re
from PIL import Image
import warnings
import cv2
from dvclive import Live

def rearange_df(df):
    """
    Rearanges the columns of the dataframe so that the order is not T12, T11, T10, T9, T8, T7, T6, T5, T4, T3, T2, T1, SC, L1, L2, L3, L4, L5, IVD_L1T12, IVD_L2L1, IVD_L3L2, IVD_L4L3, IVD_L5L4, IVD_L1, IVD_L2, IVD_L3, IVD_L4, IVD_L5
    but that its a top down approach with always the vertebrae and then the ivd below it
    Parameters:
    -----------
    df: pd.DataFrame
        dataframe with the dice scores for each label
    Returns:
    --------
    df: pd.DataFrame
        dataframe with the dice scores for each label but in the correct order
        """
    columns = df.columns
    new_T = []
    new_L = []
    ivds = [x for x in columns if "IVD" in x]
    for ivd in ivds:
        if ivd[-3] == "T" or ivd[-2]=="T":
            if ivd[-2] == "T":
                new_T.append(ivd)
                new_T.append(ivd[-2:])
            elif ivd[-3] == "T":
                new_T.append(ivd)
                new_T.append(ivd[-3:])
    new_L.append("SC")
    for ivd in ivds:
        if ivd[-2]== "L":
            new_L.append(ivd)
            new_L.append(ivd[-2:])

    
    
    new_t = np.array(new_T)[::-1]
    new_l = np.array(new_L)[::-1]
    new_columns = np.concatenate((new_t,new_l))

    df= df[new_columns]
    return df
def plot_mean_dice(transforming,summary,model,save_dir,modality,train_type):
    """"plotting function to get  a barplot of the mean dice scores for each label
    Parameters:
    -----------
    transforming: dict
        dictionary with the labels and their corresponding names
    summary: dict
        dictionary with the mean dice scores for each label
    model: str
        name of the model
    save_dir: str
        directory where the plots will be saved
    modality: str
        name of the modality
    train_type: str
        type of data that was used for training
    Returns:
    --------
    None, saves the plot in the save_dir directory"""


    mean = summary["mean"]
    individual = summary["metric_per_case"]
    df_mean = pd.DataFrame(mean)
    df_mean.rename(columns=transforming, inplace=True)

    dice_mean = rearange_df(df_mean).iloc[0]
    color_palette = []
    print(dice_mean)
    for column in dice_mean.index:
        color_palette.append("#FFA14D" if "T" in column and column not in ["T12", "IVD_L1T12"] else "#81CF6A" if "SC" in column else "#64A9F7")



    plt.figure(figsize=(15,10))
    sns_plot = sns.barplot(x=dice_mean.values, y=dice_mean.index, palette=color_palette)
    sns_plot.set_yticklabels(sns_plot.get_yticklabels(), fontsize=20)
    sns_plot.set_xticks([i/10 for i in range(11)])
    sns_plot.set_xticklabels([f'{i/10:.1f}' for i in range(11)], fontsize=20)
    plt.title(f"Mean Dice for nnU-Net: Trained on {modality}-Images with {model.split('_')[1]} Epochs, in {train_type} Case", fontsize=20,fontweight="bold") 
    plt.xlabel("Dice Coefficient [0,1]",fontsize=20)
    
    fig = sns_plot.get_figure()
    fig.savefig(os.path.join(save_dir, f"mean_dice_{model}.png"))
    return dice_mean


def plot_orccurences(df_class_dice,model,save_dir):
    """
    plotting function to get a barplot of the number of occurences for each label
    Parameters:
    -----------
    df_class_dice: pd.DataFrame
        dataframe with the dice scores for each label
    model: str
        name of the model
    save_dir: str
        directory where the plots will be saved
    Returns:
    --------
    None, saves the plot in the save_dir directory

    """

    occurences = list()
    color_palette = []
    for column in df_class_dice.columns:
        occurences.append(df_class_dice[column].count())
        color_palette.append("#FFA14D" if "T" in column and column not in ["T12", "IVD_L1T12"] else "#81CF6A" if "SC" in column else "#64A9F7")
    occurences = np.array(occurences)
    plt.figure(figsize=(15,10))
    sns_bar = sns.barplot(y=occurences, x=df_class_dice.columns, palette=color_palette)
    plt.title(f"Number of Occurences for each Label on {len(df_class_dice)} Images",fontsize=15,fontweight="bold")
    plt.xlabel("Class",fontsize=20)
    plt.ylabel("Number of Occurences",fontsize=20)
    sns_bar.set_yticklabels( sns_bar.get_yticklabels(),fontsize=10)
    sns_bar.set_xticklabels( sns_bar.get_xticklabels(),fontsize=10)
    fig = sns_bar.get_figure()
    fig.savefig(os.path.join(save_dir, f"occurences_{model}_.png"))

def plot_distplot(grouped,model,save_dir):
    """
    plotting function to get a distribution plot of the average dice scores for each image
    Parameters:
    -----------
    grouped: pd.DataFrame
        dataframe with the dice scores for each label
    model: str
        name of the model
    save_dir: str
        directory where the plots will be saved
    Returns:
    --------
    None, saves the plot in the save_dir directory
    
    """
    plt.figure(figsize=(15,10))
    all_colums_dice = grouped.columns
    #drop the reference file column
    all_colums_dice = all_colums_dice[:-1]
    grouped.replace(0, np.nan, inplace=True)
    grouped_average = grouped[all_colums_dice].mean(axis = 1,skipna=True)

    sns_plot = sns.distplot(grouped_average, bins=30, kde=False)
    plt.title(f"Distribution of Average Dice Scores for {len(grouped)} Images",fontsize=20,fontweight="bold")
    plt.xlabel("Average Dice Score [0,1]",fontsize=20)
    plt.ylabel("Number of Images",fontsize=20)
    sns_plot.set_yticklabels( sns_plot.get_yticklabels(),fontsize=20)
    sns_plot.set_xticklabels( sns_plot.get_xticklabels(),fontsize=20)
    fig = sns_plot.get_figure()
    fig.savefig(os.path.join(save_dir, f"distplot_{model}.png"))
    plt.show()

def get_semantic_dice(df_class_dice, all_colums_dice, model, dataset_result_dir):
    """
    function to get the semantic dice scores for each label
    Parameters:
    -----------
    df_class_dice: pd.DataFrame
        dataframe with the dice scores for each label
    all_colums_dice: list
        list with all the labels
    model: str
        name of the model
    dataset_result_dir: str
        directory where the plots will be saved
    Returns:
    --------
    mean_list: list
        list with the semantic dice scores for each label
    
    """
    classes = [["SC"],[x for x in all_colums_dice if "IVD" in x], [x for x in all_colums_dice if "IVD" not in x and x != "SC"]] #get all the classes
    mean_list = []
    weight_list = []
    for semantic_class in classes: #iterate over all the classes
        class_mean = 0
        total_weight = 0
        for label in semantic_class: 
            #calculate the mean dice score for each label and multiply it with the weight of the label to get the class mean
            mean_label = np.mean(df_class_dice[label].value_counts().index) 
            wegith_label = df_class_dice[label].count() / len(df_class_dice)
            class_mean += mean_label * wegith_label
            total_weight += wegith_label
        mean_list.append(class_mean)
        weight_list.append(total_weight)
    weighted_average = [mean / weight for mean, weight in zip(mean_list, weight_list)]
    return weighted_average

def _dvc_live(dice_mean, mean_list):
    """Helperfunction to log the dice scores to dvc
    Parameters:
    -----------
    dice_mean: list
        list with the dice scores for each label
    mean_list: list
        list with the semantic dice scores for each label
    Returns:
    --------
    None logs the dice scores to dvc for nnUnet"""


    with Live("nnUnet_Summary") as dvc_live:
        dvc_live.log_metric("SpinalCanal", mean_list[0])
        dvc_live.log_metric("IVD's", mean_list[1])
        dvc_live.log_metric("Vertebrae's", mean_list[2])
        dvc_live.log_metric("Mean", np.mean(dice_mean))
        if len(dice_mean) > 3:
            for label, dice in zip(dice_mean.index, dice_mean.values):
                dvc_live.log_metric(label, dice)
        
        dvc_live.make_summary("nnUnet_Summary")



def args_parser():
    parser = argparse.ArgumentParser(description='nnU-Net Summary')
    parser.add_argument('-pf', '--params_file', default='params.yaml', type=str, help='Parameters file')
    parser.add_argument('-c', '--config', default='nnU-Net_Summary', type=str, help='Config to use')
    parser.add_argument('-l', '--logs', default='INFO', type=str, help='Log level')
    parser.add_argument('-ld', '--log_dir', default='.\.logs', type=str, help='Log directory')
    parser.add_argument('-s','--save_dir', default='data/results/baseline', type=str, help='Directory where results will be saved')
    parser.add_argument('-d', '--dataset', default='data/nnUNet_trained_models/Dataset004_SPIDER', type=str, help='Summary of the dataset')
    parser.add_argument('-t', '--train_type', default='2d', type=str, help='Type of data that was used for training')
    return parser.parse_args()



if __name__ == '__main__':
    # Ignore all warnings
    warnings.filterwarnings("ignore")       
    args = args_parser()
    kwargs = load_yaml(args.params_file)[args.config]
    logger = set_logger(name = "summary baseline",save_path= args.log_dir, log_level= args.logs)

    if not os.path.exists(kwargs.get("save_dir",args.save_dir)):
        os.makedirs(kwargs.get("save_dir",args.save_dir))
    #get to the summary, json file

    save_dir = kwargs.get("save_dir",args.save_dir)


    #get the dataset from the pipeline_summary.yaml file
    pipeline_yaml_path = kwargs.get("pipeline_yaml", "config/data/pipeline_summary.yaml")
    pipeline_yaml = load_yaml(pipeline_yaml_path)
    dataset = pipeline_yaml["Dataset"]

    nnunet_raw_dir = kwargs.get("nnunet_raw", "data/nnUNet_raw")

    dataset_result_dir = os.path.join(nnunet_raw_dir, dataset)


    train_type = args.train_type
    pipeline_yaml["train_type"] = train_type
    save_yaml(pipeline_yaml, pipeline_yaml_path)

    if not os.path.exists(dataset_result_dir):
        os.makedirs(dataset_result_dir)
    #get to the summary, json file
    dataset = dataset_result_dir.split("/")[-1]

    assert str(dataset).startswith("Dataset"), "The dataset directory should start with Dataset"

    dataset_json_path = os.path.join("data/nnUNet_raw",dataset, "dataset.json")

    #extract the labels
    with open(dataset_json_path) as f:
        dataset_json = json.load(f)
        labels = dataset_json["labels"]
        modality = list(dataset_json["channel_names"].keys())[0]
    transforming = {str(v):k for k,v in labels.items()}

    all_trainings = os.listdir(dataset_result_dir)
    print(all_trainings)
    print(dataset)
    print(dataset_result_dir)
    models_trained = [x for x in all_trainings if x.split("_")[-1]==train_type]

    print(models_trained)
    assert models_trained, "No models trained with this type of data"
    logger.info(f"models trained with this type of data:{train_type} and with this dataset:{dataset} on this modality: {modality}")
    for model in models_trained: #plots average dice score for each model
        print(model)
        whole_path_summary = os.path.join(dataset_result_dir, model,'crossval_results_folds_0_1_2_3_4', "summary.json")
        print(whole_path_summary)

        with open(whole_path_summary) as f:
            summary = json.load(f)
        mean = summary["mean"]
        individual = summary["metric_per_case"]

        dice_mean = plot_mean_dice(transforming,summary,model,save_dir ,modality,train_type)
        df_individual = pd.DataFrame(individual)


    
        class_dice_data = []
        for index, row in df_individual.iterrows():
            # Extract the metrics dictionary for the current row
            metrics = row['metrics']
            
            # Initialize a dictionary to store the "Dice" values for each class
            class_dice = {}
            
            # Iterate through each class in the metrics dictionary
            for class_key, class_data in metrics.items():
                class_dice[class_key] = class_data['Dice']
            
            # Append the class_dice dictionary to the class_dice_data list
            class_dice_data.append(class_dice)

        # Create a new DataFrame from the class_dice_data list
        df_class_dice = pd.DataFrame(class_dice_data)
        df_class_dice.rename(columns=transforming, inplace=True)
        df_class_dice = rearange_df(df_class_dice)
        plot_orccurences(df_class_dice,model,save_dir)
        all_colums_dice = df_class_dice.columns
        df_class_dice["reference_file"]= df_individual["reference_file"]
        #gets us the best average dice score for each image and worst average dice score for each image
        #df.replace(0, np.nan, inplace=True) maybe use this to replace all zeros with nans and then we can filter out nans to get overall average dice score
        df_class_dice.replace(0, np.nan, inplace=True)
        
        mean_list = get_semantic_dice(df_class_dice, all_colums_dice, model, save_dir)

        logger.info(f"mean dice score for SC: {mean_list[0]}, mean dice score for IVD: {mean_list[1]}, mean dice score for vertebrae: {mean_list[2]}")
        grouped = df_class_dice.groupby(df_class_dice[all_colums_dice].mean(axis = 1,skipna=True)).max()

        plot_distplot(grouped,model,save_dir)
        min_score = grouped.iloc[0]
        max_score = grouped.iloc[-1]

        logger.info(f"min score: {min_score[all_colums_dice].mean(skipna= True)}\nmax score: {max_score[all_colums_dice].mean(skipna= True)}")
        logger.info(f'For the semantic dice we get a weighted average for:\n Spinal Canal: {mean_list[0]:.04f}\nfor IVD: {mean_list[1]:.04f}\nfor Vertebrae: {mean_list[2]:.04f}')

        _dvc_live(dice_mean, mean_list) #log the dice scores to dvc

        for score, name in zip([min_score, max_score], ["minimal score", "maximal score"]):
            label = score["reference_file"]
            image = re.sub('labelsTr', "imagesTr", label)
            label_pred = re.sub(f'nnUnet_raw/{dataset}/labelsTr',
                                f"nnUNet_trained_models/{dataset}/{model}/crossval_results_folds_0_1_2_3_4/postprocessed/labelsTr", label)

            image_data = nib.load(image.replace(".nii.gz", "_0000.nii.gz")).get_fdata()
            label_data = nib.load(label).get_fdata()
            label_data_pred = nib.load(label_pred).get_fdata()

            image_central_slice = image_data[image_data.shape[0]//2, :, :]
            label_central_slice = label_data[label_data.shape[0]//2, :, :]
            label_central_slice_pred = label_data_pred[label_data_pred.shape[0]//2, :, :]

            class_colors = {}
            for class_label in np.unique(label_central_slice):
                class_colors[class_label] = tuple(np.random.randint(0, 256, size=3))

            # Convert images to uint8
            image_central_slice = (image_central_slice * 255).astype(np.uint8)
            label_central_slice = (label_central_slice * 255).astype(np.uint8)
            label_central_slice_pred = (label_central_slice_pred * 255).astype(np.uint8)

            canvas = np.zeros_like(image_central_slice, dtype=np.uint8)
            canvas_pred = np.zeros_like(image_central_slice, dtype=np.uint8)

            for class_label, color in class_colors.items():
                # Extract RGB components from the color tuple
                color_np = np.array(color, dtype=np.uint8)

                # Create boolean masks for the specified class label
                mask = (label_central_slice == class_label)
                mask_pred = (label_central_slice_pred == class_label)

                # Assign RGB values to the corresponding positions in the canvas array
                canvas[mask] = color_np[0]  # For grayscale images, use only the first component
                canvas_pred[mask_pred] = color_np[0]

            result_lab = cv2.addWeighted(image_central_slice, 0.2, canvas, 0.8, 0)
            result_lab_pred = cv2.addWeighted(image_central_slice, 0.2, canvas_pred, 0.8, 0)
            result_merged = cv2.addWeighted(canvas_pred, 0.2, canvas, 0.8, 0)

            result_lab = result_lab.astype(np.uint8)
            result_lab_pred = result_lab_pred.astype(np.uint8)
            result_merged = result_merged.astype(np.uint8)

            result_lab_image = Image.fromarray(result_lab)
            result_lab_pred_image = Image.fromarray(result_lab_pred)
            result_merged_image = Image.fromarray(result_merged)

            result_lab_image.save(os.path.join(dataset_result_dir, f"{label.split('/')[-1]}_result_lab.png"))
            result_lab_pred_image.save(os.path.join(dataset_result_dir, f"{label.split('/')[-1]}_result_lab_pred.png"))
            result_merged_image.save(os.path.join(dataset_result_dir, f"{label.split('/')[-1]}_result_merged.png"))


            

    
