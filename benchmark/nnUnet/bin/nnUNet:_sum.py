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





def args_parser():
    parser = argparse.ArgumentParser(description='nnU-Net Summary')
    parser.add_argument('-pf', '--params_file', default='params.yaml', type=str, help='Parameters file')
    parser.add_argument('-c', '--config', default='nnU-Net_Summary', type=str, help='Config to use')
    parser.add_argument('-l', '--logs', default='INFO', type=str, help='Log level')
    parser.add_argument('-ld', '--log_dir', default='.\.logs', type=str, help='Log directory')
    parser.add_argument('-s','--save_dir', default='data/results/baseline', type=str, help='Directory where results will be saved')
    parser.add_argument('-d', '--dataset', default='Dataset010_SPIDER', type=str, help='Summary of the dataset')
    parser.add_argument('-t', '--train_type', default="fullres", type=str, help='Type of data that was used for training')
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
    dataset = args.dataset

    nnunet_raw_dir = kwargs.get("nnunet_raw", "data/nnUNet_raw")

    dataset_result_dir = os.path.join(nnunet_raw_dir, dataset)
    
    
    train_type = args.train_type
    train_type = "2d"
    train_type_sum = train_type
    print(train_type_sum)
    if train_type == "fullres":
        train_type_sum = "3d_fullres"
    elif train_type == "2d":
        train_type_sum = "2d"
 
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
        modality = list(dataset_json["channel_names"].keys())[0]




    dataset_result_path = str.replace(dataset_result_dir, "nnUNet_raw", "nnUNet_trained_models")
    dataset_result_dir = os.path.join(dataset_result_path, f"nnUNetTrainer_250epochs_nnUNetPlans__{train_type_sum}")
    all_trainings = os.listdir(dataset_result_path)
    print(all_trainings)
    print(dataset)
    print(dataset_result_dir)
    models_trained = [x for x in all_trainings if x.split("_")[-1]==train_type]

    print(models_trained)
    assert models_trained, "No models trained with this type of data"
    logger.info(f"models trained with this type of data:{train_type} and with this dataset:{dataset} on this modality: {modality}")
    for model in models_trained: #plots average dice score for each model
        print(model)
        whole_path_summary = os.path.join(dataset_result_path, model,'crossval_results_folds_0_1_2_3_4', "summary.json")
        print(whole_path_summary)

        with open(whole_path_summary) as f:
            summary = json.load(f)
        mean = summary["mean"]


        individual = summary["metric_per_case"]


        #make a df for individual
        df = pd.DataFrame(individual)
        print(df.head)
        #print first row
        print(df.iloc[0])

# Assuming your dataframe is named df
# Extracting Dice Scores for labels 1, 2, and 3
        df['Dice_1'] = df['metrics'].apply(lambda x: x.get('1', {}).get('Dice', None))
        df['Dice_2'] = df['metrics'].apply(lambda x: x.get('2', {}).get('Dice', None))
        df['Dice_3'] = df['metrics'].apply(lambda x: x.get('3', {}).get('Dice', None))

        # Drop the original 'metrics' column if needed
        df.drop(columns=['metrics'], inplace=True)
        ids = [1, 50, 58, 130, 133, 152, 160, 186, 251]

    # Generate the formatted strings for these IDs
        str_id = ['SPIDER_{:03d}'.format(i) for i in ids]
        df = df[~df['prediction_file'].str.contains('|'.join(str_id))]

        # Display the updated dataframe
        print(df.head())
        print(df.describe())

        #iterate over the prediction column and save the image as a png
        for row in df["prediction_file"]:
            label = nib.load(row).get_fdata()
            image_original_name = row.split("/")[-1].split(".")[0]
            image_original_name += "_0000.nii.gz"
            orignal_image_path = os.path.join(nnunet_raw_dir, dataset, "imagesTr", image_original_name)
            original_image = nib.load(orignal_image_path).get_fdata()

            original_image = original_image[original_image.shape[0]//2, :, :]



            print(row)

            if len(label.shape) ==3:
                onehot_encoded = np.zeros((3, label.shape[0], label.shape[1], label.shape[2]))
            else:
                onehot_encoded = np.zeros((3, label.shape[0], label.shape[1]))

            # One-hot encode classes 1, 2, and 3 while excluding class 0
            for i in range(1, 4):
                onehot_encoded[i-1][label == i] = 1
            print(onehot_encoded.shape)
            for channel in range(onehot_encoded.shape[0]):
                if len(onehot_encoded.shape) == 3:
                    img = onehot_encoded[channel, :, :,]
                else:
                    #take the middle slice
                    img = onehot_encoded[channel, onehot_encoded.shape[1]//2, :, :]
                #save each image as a png
                #save the image

                img_save_path = os.path.join(save_dir,dataset, model, str(channel))
                img_name = row.split("/")[-1].split(".")[0]
                if not os.path.exists(img_save_path):
                    os.makedirs(img_save_path)
                img_path = os.path.join(img_save_path, f"{img_name}.png")
                #save img
                plt.imsave(img_path, img, cmap="gray")

                
            save_dir_masks = os.path.join(save_dir, dataset, model, "masks")
         
            img = onehot_encoded[:, onehot_encoded.shape[1]//2, :, :]

            current_seg_np = img

                                            
    
            combined_mask = np.zeros((320, 320, 3), dtype=np.uint8)

            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] 
            for i in range(3):
                mask = current_seg_np[i]
                colored_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2) * colors[i]
                combined_mask = cv2.add(combined_mask, colored_mask.astype(np.uint8))
            print(combined_mask.shape)

            #save the combined mask img as a png
            
            os.makedirs(save_dir_masks, exist_ok=True)  # Create directory if it doesn't exist

            cv2.imwrite(os.path.join(save_dir_masks,f'combined_mask_{img_name}.png'),combined_mask)

            
            print(original_image.shape, "original image shape")

            original_image = original_image.astype(np.uint8)
            alpha = 0.75
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
            blended_image = cv2.addWeighted(original_image, 1, combined_mask, alpha, 0)

            mask_dir = os.path.join(save_dir, dataset, model, "blended_image")
            os.makedirs(mask_dir, exist_ok=True)  # Create directory if it doesn't exist

            cv2.imwrite(os.path.join(mask_dir,f'blended_image_{img_name}.png'),blended_image)

        for row in df["reference_file"]:

            image_path = str.replace(row, "labelsTr", "imagesTr")
            image_n = image_path.split("/")[-1].split(".")[0]
            image_n += "_0000.nii.gz"

            image_path = str.replace(image_path, image_path.split("/")[-1], image_n)

            print(image_path)

            image = nib.load(image_path).get_fdata()
            #save the image
            imgage_save_path = os.path.join(save_dir, dataset, model, "image")
            if not os.path.exists(imgage_save_path):
                os.makedirs(imgage_save_path)
            img_path = os.path.join(imgage_save_path, f"{image_n.split('.')[0]}.png")

            image = image[image.shape[0]//2, :, :]
            plt.imsave(img_path, image, cmap="gray")

            image = image.astype(np.uint8)

            





            true_label = nib.load(row).get_fdata()
            if len(true_label.shape) ==3:
                onehot_encoded = np.zeros((3, true_label.shape[0], true_label.shape[1], true_label.shape[2]))
            else:
                onehot_encoded = np.zeros((3, true_label.shape[0], true_label.shape[1]))
            
            for i in range(1, 4):
                onehot_encoded[i-1][true_label == i] = 1
            print(onehot_encoded.shape)

            for channel in range(onehot_encoded.shape[0]):
                if len(onehot_encoded.shape) == 3:
                    img = onehot_encoded[channel, :, :,]
                else:
                    #take the middle slice
                    img = onehot_encoded[channel, onehot_encoded.shape[1]//2, :, :]
                #save each image as a png
                #save the image
                print(img.shape)
                img_save_path = os.path.join(save_dir, dataset, model, str(channel))
                img_name = row.split("/")[-1].split(".")[0]
                if not os.path.exists(img_save_path):
                    os.makedirs(img_save_path)
                img_path = os.path.join(img_save_path, f"original_{img_name}.png")
                #save img
                plt.imsave(img_path, img, cmap="gray")

            save_dir_masks = os.path.join(save_dir, dataset, model, "masks_original")
         
            img = onehot_encoded[:, onehot_encoded.shape[1]//2, :, :]

            current_seg_np = img

                                            

            combined_mask = np.zeros((320, 320, 3), dtype=np.uint8)

            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] 
            for i in range(3):
                mask = current_seg_np[i]
                colored_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2) * colors[i]
                combined_mask = cv2.add(combined_mask, colored_mask.astype(np.uint8))
            print(combined_mask.shape)

            #save the combined mask img as a png
            
            os.makedirs(save_dir_masks, exist_ok=True)  # Create directory if it doesn't exist

            cv2.imwrite(os.path.join(save_dir_masks,f'combined_mask_{img_name}.png'),combined_mask)

 

            original_image = image
            alpha = 0.75
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
            blended_image = cv2.addWeighted(original_image, 1, combined_mask, alpha, 0)

            mask_dir = os.path.join(save_dir, dataset, model, "blended_image_original")
            os.makedirs(mask_dir, exist_ok=True)  # Create directory if it doesn't exist

            cv2.imwrite(os.path.join(mask_dir,f'blended_image_{img_name}.png'),blended_image)


        