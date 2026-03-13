import logging
from pathlib import Path
import torch
import os
import argparse
import numpy as np
from typing import List, Tuple
from monai.utils import set_determinism
from src.dataset.DataloaderSPIDER import get_dataloader_SPIDER
from src.results.metric import dice
from src.networks.LumbarSpineSegDiff import LumbarSpineSegDiffInference
from monai.data import DataLoader
from skimage.morphology import remove_small_objects
from scipy import ndimage as ndi

import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
set_determinism(123)

def maximum_entropy_aggregation(uncertainty_timesteps: torch.Tensor) -> torch.Tensor:
    """
        Compute the variance-weighted aggregation of entropy heatmaps across timesteps,
        preserving spatial information and rescaling the output to create a combined entropy map.

        Parameters:
        -----------
        uncertainty_timesteps: torch.Tensor  (n_timesteps, 1, seg_channels, height, width)
            Stack of entropy heatmaps for each timestep of shape:

        Returns:
        --------
        H_combined_map_rescaled: torch.Tensor
            The combined and rescaled entropy map. Shape: (seg_channels, height, width)
    """
    # Compute the maximum entropy across time-steps for each spatial location and channel
    H_max = torch.max(uncertainty_timesteps, dim=0)[0] # Shape: (seg_channels, height, width)

    return H_max


def variance_weighted_entropy_aggregation_map(uncertainty_timesteps):
    """
        Compute the variance-weighted aggregation of entropy heatmaps across timesteps,
        preserving spatial information and rescaling the output to create a combined entropy map.

        Parameters:
        -----------
        uncertainty_timesteps: torch.Tensor 15, 1, 3, 320, 320
            Stack of entropy heatmaps for each timestep of shape:
            (n_timesteps, 1, seg_channels, height, width)

        Returns:
        --------
        H_combined_map_rescaled: torch.Tensor
            The combined and rescaled entropy map. Shape: (seg_channels, height, width)
    """
    # Store original min and max values
    original_min = uncertainty_timesteps.min()
    original_max = uncertainty_timesteps.max()

    # Normalize input to [0, 1] range
    normalized_uncertainty = (uncertainty_timesteps - original_min) / (original_max - original_min)

    # Compute variance across timesteps for each spatial location and channel
    variances = torch.var(normalized_uncertainty, dim=0)  # Shape: (1, seg_channels, height, width)

    # Normalize variances to use as weights
    weights = variances / torch.sum(variances)

    # Compute the weighted sum of normalized entropy heatmaps across timesteps
    H_combined_map = torch.sum(weights.unsqueeze(0) * normalized_uncertainty, dim=0)  # Shape: (seg_channels, height, width)

    # # Rescale the output to the original range
    H_combined_map_rescaled = H_combined_map * (original_max - original_min) + original_min

    return H_combined_map_rescaled

class LumbarSpineSegDiffInferer():
    def __init__(self, env_type, batch_size, device="cpu",  num_gpus=1, logdir="./logs/", num_timesteps = 1000, presegmentation = False, ensemble = 5, Ts =15):
        assert env_type in ["pytorch", "cuda", "gpu"], f"not support this env_type: {env_type}"
        self.env_type = env_type
        self.num_gpus = num_gpus
        self.device = device
        self.batch_size = batch_size
        self.logdir = logdir
        self.num_timesteps = num_timesteps
        self.presegmentation = presegmentation
        self.num_ensemble = ensemble
        self.model = LumbarSpineSegDiffInference(self.num_timesteps,
                                           self.presegmentation,
                                           modality=1,
                                           num_targets=3,
                                           skip_timesteps=50,
                                           device=self.device,
                                           weights_path=self.logdir + "/weights",
                                           samples_timestep = ensemble,
                                           Ts = Ts
                                           )


    def test_batch(self, datalist, save_path=None):
        """test for single gpu model
        Parameters:
        -----------
        val_dataset: Dataset
            the validation dataset
        save_path: str
            the path to save the output and uncertainty
        Returns:
        --------
        v_sum: float
            the mean of the validation output
        val_outputs: list
            the list of the validation output
        saves the output and uncertainty to the save_path
        """

        dataloader = DataLoader(datalist, batch_size=1, shuffle=False)
        self.model.to(self.device)
        val_outputs = []
        self.model.eval()
        for idx, (batch, image_filepath) in tqdm(enumerate(dataloader), total=len(dataloader)):

            image, label, label_pre = self.get_input_batch(batch)

            with torch.no_grad():
                output_scores, output, uncertainty_map = self.test_step(image, label, label_pre=label_pre)

                # Save the output and uncertainty
                if save_path is not None:
                    self.save_channelwise_output_mask(output, image_filepath, save_path)

                    # variance_map, variance_map_normalized = self.save_variance_uncertainty(save_path, image_filepath, uncertainty_map, cmap="jet")
                    self.save_uncertainty_map(uncertainty_map, save_path, image_filepath, cmap="jet")

                    self.save_overlay_label_mask(image, label, image_filepath,  save_path)
                    self.save_overlay_results_mask(image, output, image_filepath, save_path)

                    self.save_error_map(image_filepath, label, output, save_path)


            self.save_scores(image_filepath, save_path, output_scores)

            val_outputs.append(output_scores)

        v_sum, val_outputs = self.compute_mean_score(output_scores, val_outputs)
        return v_sum, val_outputs

    def save_error_map(self, image_filepath, label, output, save_path):
        try:
            error_map = np.abs(np.squeeze(label.cpu().numpy()[0, :, :, :]) - np.squeeze(output[:, :, :]))

            logging.info("The Error Map shape is: ",error_map.shape)
            save_filepath = Path(save_path) / "error-maps" / f'errormap_{image_filepath[0].split(".")[0]}.png'
            Path(save_filepath.parent).mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
            plt.imsave(save_filepath, np.sum(error_map, axis=0), cmap='viridis')
        except:
            logging.error("Error in saving error map")

    def save_overlay_results_mask(self, image, output, image_filepath,  save_path):

        output_mask_dir = Path( save_path)/"masks"  #os.path.join(save_path, "masks")

        outout_color_mask = self.create_color_mask(output[0, :, :, :], fileaname=image_filepath[0].split(".")[0], mask_dir=output_mask_dir.as_posix())

        alpha = 0.1
        image_array = self.normalize_image( image.squeeze().cpu().numpy())
        # save the image as a png
        blended_image = cv2.addWeighted(image_array, 1 - alpha, outout_color_mask, alpha, 0)
        cv2.imwrite(os.path.join(output_mask_dir, f'masks_image_{image_filepath[0].split(".")[0]}.png'), blended_image)



    def save_overlay_label_mask(self,  image, label, image_filepath, save_path):
        alpha = 0.2
        image_array = self.normalize_image(image.squeeze().cpu().numpy())
        labels_dir = Path(save_path) / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        gt_color_mask = self.create_color_mask(label.cpu().numpy()[0, :, :, :])
        label_image = cv2.addWeighted(image_array, 1 - alpha, gt_color_mask, alpha, 0)
        filename = labels_dir / f'label_image_{image_filepath[0].split(".")[0]}.png'
        cv2.imwrite(filename.as_posix(), label_image)

    def compute_mean_score(self, output_scores,  val_outputs):
        return_list = False
        if isinstance(output_scores, list) or isinstance(output_scores, tuple):
            return_list = True
        val_outputs = torch.tensor(val_outputs).to(self.device)
        if not return_list:
            length = 0
            v_sum = 0.0
            for v in val_outputs:
                if not torch.isnan(v).any():
                    v_sum += v
                    length += 1

            if length == 0:
                v_sum = 0
            else:
                v_sum = v_sum / length
        else:
            num_val = len(val_outputs[0])
            length = [0.0 for i in range(num_val)]
            v_sum = [0.0 for i in range(num_val)]

            for v in val_outputs:
                for i in range(num_val):
                    if not torch.isnan(v[i]).any():
                        v_sum[i] += v[i]
                        length[i] += 1

            for i in range(num_val):
                if length[i] == 0:
                    v_sum[i] = 0
                else:
                    v_sum[i] = v_sum[i] / length[i]
        return v_sum, val_outputs

    def save_channelwise_output_mask(self, output, path, save_path):
        os.makedirs(os.path.join(save_path, "output"), exist_ok=True)
        saving_path = os.path.join(save_path, "output", path[0].split(".")[0])
        for channel in range(output.shape[1]):
            img = output[0, channel, :, :]
            assert len(img.shape) == 2
            plt.imsave(saving_path + f"output_{channel}.png", img, cmap='gray')

    def save_scores(self, image_filepath, save_path, scores,  header= ["image", "SC", "VB", "IVD"]):

        # create a csv file with the dics scores results for each individual image
        csv_path = os.path.join(save_path, "dice_semantic.csv")

        if not os.path.exists(csv_path):
            with open(csv_path, "w") as file:
                file.write(",".join(header) + "\n")

        with open(csv_path, "a") as file:
            scores_str = f"{image_filepath[0].split('.')[0]},{','.join(map(str, scores))}"
            file.write(f"{scores_str}\n")
            #file.write(f"{image_filepath[0].split('.')[0]},{scores[0]},{scores[1]},{scores[2]}\n")


    def get_input_batch(self, batch):
        """
        Get the input batch for the model.

        Parameters:
        -----------
        batch : dict, list, or torch.Tensor
            The batch of data. It can be a dictionary, list, or tensor.

        Returns:
        --------
        tuple
            A tuple containing the image tensor, label tensor, and optionally the pre-segmentation label tensor.
        """

        if isinstance(batch, dict):
            batch = {
                x: batch[x].to(self.device)
                for x in batch if isinstance(batch[x], torch.Tensor)
            }
        elif isinstance(batch, list):
            batch = [x.to(self.device) for x in batch if isinstance(x, torch.Tensor)]

        elif isinstance(batch, torch.Tensor):
            batch = batch.to(self.device)

        else:
            print("not support data type")
            exit(0)

        # Get the inputs
        image = batch["image"].float()
        label = batch["label"].float()
        label_pre =  batch["label_pre"].float() if self.presegmentation else None

        return image, label,  label_pre


    def normalize_image(self, current_img_np):
        # normalize img
        current_img_np = (current_img_np - current_img_np.min()) / (current_img_np.max() - current_img_np.min())
        current_img_np = (current_img_np * 255).astype(np.uint8)
        current_img_np = cv2.cvtColor(current_img_np, cv2.COLOR_GRAY2BGR)
        return current_img_np


    def create_color_mask(self, output_mask, fileaname="", mask_dir=""):
        color_mask = np.zeros((320, 320, 3), dtype=np.uint8)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for i in range(3):
            mask = output_mask[i]
            colored_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2) * colors[i]
            color_mask = cv2.add(color_mask, colored_mask.astype(np.uint8))

        if fileaname or mask_dir:
            os.makedirs(mask_dir, exist_ok=True)  # Create directory if it doesn't exist
            # save the combined mask img as a png
            cv2.imwrite(os.path.join(mask_dir, f'mask_{fileaname}.png'), color_mask)
        return color_mask



    def save_uncertainty_map(self, uncertainty_heatmap, imgage_dir, path, cmap="jet"):
        """
        Save the variance uncertainty maps as images.

        Parameters:
        -----------
        imgage_dir: str
            The directory to save the variance uncertainty maps.
        path: str
            The path of the input data.
        var: torch.Tensor
            The variance uncertainty map.

        Returns:
        --------
        tuple
            A tuple containing the variance map and the normalized variance map.
        """

        uncertainty_heatmap = uncertainty_heatmap.squeeze(0).cpu().numpy()
        save_filepath = Path(imgage_dir) / "uncertainty_maps"/ f'uncertainty_heatmap_{path[0].split(".")[0]}.png'
        Path(save_filepath.parent).mkdir(parents=True, exist_ok=True)
        plt.imsave( save_filepath.as_posix(),   np.sum(uncertainty_heatmap, axis=0), cmap=cmap)


    def save_variance_uncertainty(self, imgage_dir, path, uncertainty_heatmap, cmap="jet"):
        """
        Save the variance uncertainty maps as images.

        Parameters:
        -----------
        imgage_dir: str
            The directory to save the variance uncertainty maps.
        path: str
            The path of the input data.
        var: torch.Tensor
            The variance uncertainty map.

        Returns:
        --------
        tuple
            A tuple containing the variance map and the normalized variance map.
        """

        uncertainty_heatmap = uncertainty_heatmap.squeeze(0).cpu().numpy()
        variance_map_normalized = np.zeros_like(uncertainty_heatmap)
        combined_variance_map_sum = np.sum(uncertainty_heatmap, axis=0)
        combined_variance_map_mean = np.mean(uncertainty_heatmap, axis=0)
        combined_variance_dir = os.path.join(imgage_dir, "uncertainty_maps")
        Path(combined_variance_dir).mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

        plt.imsave(
            os.path.join(combined_variance_dir, f'uncertainty_mapmean_{path[0].split(".")[0]}.png'),
            combined_variance_map_mean, cmap=cmap)
        return uncertainty_heatmap, variance_map_normalized


    def load_state_dict(self, weight_path, strict=True):
        """Load the state dict of the model
        Parameters:
        ----------
        weight_path: str
            The path of the state dict
        strict: bool
            The strict of the model
        """
        sd = torch.load(weight_path, map_location="cpu")
        if "module" in sd :
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k
            new_sd[new_k] = v

        self.model.load_state_dict(new_sd, strict=strict)
        logging.info(f"model parameters are loaded successfully")

    def post_process_masks(self, masks, min_size=12):
        """
        Post-process segmentation masks by removing small objects and filling holes

        Args:
            masks: Tensor of shape (B, C, H, W) with binary masks
            min_size: Minimum size of objects to keep
            hole_size: Maximum size of holes to fill

        Returns:
            Processed masks tensor of same shape
        """
        # check if mask is a tensor or numpy array
        if  isinstance(masks, torch.Tensor):
            masks_array = masks.detach().cpu().numpy()
        else:
            # Make a copy of the masks
            masks_array = masks.copy()


        B, C, H, W = masks_array.shape

        # Process each batch and channel
        processed_mask = np.zeros_like(masks_array)

        for b in range(B):
            max_mask= np.max(masks_array[b], axis=0)

            for c in range(C):

                # Prioritize the mask with the of IVD over the other masks
                mask = masks_array[b, c].copy()
                # m = np.where(mask == max_mask, mask, 0)
                #
                # mask =  np.where( masks_array[b, c].copy() == max_mask,  masks_array[b, c], 0)


                # # Fill small holes
                # m = remove_small_holes(m, area_threshold=min_size)

                # Fill all holes within the mask
                m = ndi.binary_fill_holes(mask.astype(bool))

                # m = binary_dilation(m, disk(1))

                # Remove small objects
                m = remove_small_objects(m.astype(bool),
                                               min_size=min_size//4 if (c%2) else min_size,
                                               connectivity=2)

                m = ndi.binary_fill_holes(m)
                m = ndi.binary_fill_holes(m)

                m = remove_small_objects(m.astype(bool),
                                         min_size=min_size // 4 if (c % 2) else min_size,
                                         connectivity=2)

                # m = clear_border(m)
                #
                # # # Erode the mask by 1 pixel
                # m = erosion(m, disk(1))

                processed_mask[b, c] =m

        if isinstance(masks, torch.Tensor):
            # Convert back to tensor
            processed_mask = torch.from_numpy(processed_mask)
            if masks.is_cuda:
                processed_mask = processed_mask.cuda()

        return processed_mask



    def test_step(self, image, label, label_pre=None):
        """The test step of the model
        Parameters:
        ----------
        batch: dict
            The batch of the data
        path: str
            The path to save the output
        Returns:
        -------
        list
            The dice score of the model
        """

        with torch.no_grad():
            x = label if not self.presegmentation else label_pre

            output, uncertainty_maps = self.model(x=x,  image=image, step=self.num_timesteps,
                                                 pred_type="ddim_sample")

            uncertainty = maximum_entropy_aggregation(torch.stack(uncertainty_maps))

            output = torch.sigmoid(output)#, dim=1)

            # Binarize the output
            output = (output > 0.5).float()#.cpu().numpy().astype(int)
            #Check if the output is mutually exclusive across channels
            # # Assuming 'output' is your tensor of shape [1, 3, 320, 320]
            # output = torch.argmax(torch.softmax(output, dim=1) , dim=1)  # Convert logits to probabilities
            # output = torch.nn.functional.one_hot(output, num_classes=3).permute(0, 3, 1, 2).float()

            output = output.cpu().numpy().astype(int)

            output = self.post_process_masks(output, min_size=64)

            #
            # # Compute the softmax such that the segmentation mask is mutually exclusives across channels
            # output = output / np.sum(output, axis=1, keepdims=True)
            # output = torch.from_numpy(output).to(self.device)



            dice_scores = self.compute_bach_dice(label, output)

            logging.info(f"Validation Dice Scores of the model: {dice_scores}")

        return dice_scores, output, uncertainty

    def compute_bach_dice(self, label: torch.Tensor, output: torch.Tensor) ->  Tuple[List[float], np.ndarray]:
        """
        Compute the Dice score for each channel in the output.

        Parameters
        ----------
        label : torch.Tensor
            The ground truth labels.
        output : torch.Tensor
            The predicted output from the model.

        Returns
        -------
        Tuple[List[float], np.ndarray]
            A tuple containing the list of Dice scores for each channel and the binarized output.
        """

        target = label.cpu().numpy()
        dice_scores = list()
        for i in range(output.shape[1]):

            o = output[:, i]
            t = target[:, i]
            assert len(np.unique(0)) <= 2 and len(np.unique(t)) <= 2
            dice_scores.append(dice(o, t))
        return dice_scores


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--data_dir", type=str, default="./data/SPIDER_T2w")
    parser.add_argument("-dev","--device", type=str, default="cuda:0")
    parser.add_argument("-c","--num_classes", type=int,default=4)
    parser.add_argument("-w","--weights_dir", type=str,default= "./results/LumbarSpineSegDiff/SPIDER_T1wT2w/fold-0/")
    parser.add_argument("-f","--fold", type=int,default= 0)
    parser.add_argument("-p","--presegmentation",action='store_true')
    parser.add_argument("-e","--num_ensemble", type=int,default= 5)
    parser.add_argument("-ts","--timsteps_sample", type=int,default= 15)
    parser.add_argument("-t","--timesteps", type=int,default= 1000)
    parser.add_argument("-sp","--save_path", type=str,default= "./results/LumbarSpineSegDiff/SPIDER_T1wT2w/fold-0/visualization")
    args = parser.parse_args()
    return args


if __name__ == "__main__":


    args = arg_parser()
    data_dir = args.data_dir
    split_path = os.path.join(data_dir, "splits_final.json")
    batch_size = 1
    device = args.device
    fold = args.fold
    number_targets = args.num_classes - 1
    save_path = args.save_path
    num_classes = args.num_classes
    num_timesteps = args.timesteps
    presegmentation = args.presegmentation
    Ts = args.timsteps_sample
    weights_dir = args.weights_dir

    print(f"Data directory: {data_dir}\n"
      f"Fold: {fold}\n"
      f"Number of targets: {number_targets}\n"
      f"Number of ensemble: {args.num_ensemble}\n"
      f"Presegmentation: {presegmentation}\n"
      f"Save path: {save_path}\n"
      f"Number of timesteps: {num_timesteps}\n"
      f"Weights directory: {weights_dir}")

    train_ds, val_ds, test_ds = get_dataloader_SPIDER(data_dir=Path(data_dir),
                                                      split_path=split_path,
                                                      num_classes=num_classes,
                                                      fold=args.fold,
                                                      presegmentation=presegmentation,
                                                      presegmentation_dir = "./results/nnUnet/SPIDER_T1w_T2w/outputs")

    predictor = LumbarSpineSegDiffInferer(env_type="pytorch",
                                    batch_size=batch_size,
                                    device=device,
                                    num_gpus=1,
                                    num_timesteps =num_timesteps,
                                    presegmentation= presegmentation,
                                    ensemble=args.num_ensemble,
                                    Ts = Ts)

    models = os.listdir(os.path.join(weights_dir, "weights"))
    print("Load weights from: " , models)
    best_model = 0
    best_model_path = list(Path(weights_dir, "weights").glob("best_*.pt"))
    if best_model_path:
        best_model_path = list(Path(weights_dir, "weights").glob("final_*.pt"))
    else:
        best_model_path = [str(path) for path in Path(weights_dir, "weights").glob("*.pt")]

    if not best_model_path:
        logging.error(f"No model found in folder {weights_dir} check the train.py file")
    else :

        print(f''
              f'The best model was: {best_model_path[0].stem}')
        predictor.load_state_dict(best_model_path[0].as_posix(), strict=False)
        v_mean, _ = predictor.test_batch(datalist=test_ds, save_path = save_path)
        print(f" The mean of the validation output is:  {[m.cpu().numpy() for m in v_mean ] }")