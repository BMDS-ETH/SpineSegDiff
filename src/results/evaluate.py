IDS = [162, 186, 85, 82, 181, 38, 10, 65, 226, 23]
import torch
from pathlib import Path
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
from src.dataset.DataloaderSPIDER import get_dataloader_SPIDER
from monai.data import DataLoader
from PIL import Image, ImageOps

def make_grid(images, rows, cols):

    imgs = [im.rotate(90, expand=True)  for im in images]
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*(w+10), rows*(h+10)))
    for i, image in enumerate(imgs):
        grid.paste( ImageOps.expand(image, border=5, fill='white'), box=(i%cols*w, i//cols*h))
        #grid.paste( ImageOps.expand(image, border=5, fill='white'), box=(i%cols*h, i//cols*w))

        #grid.paste(image, box=(i%cols*w, i//cols*h))
        #grid.paste( ImageOps.expand(image, border=5, fill='white'), box=(i//cols*w, i%cols*w))
    return grid

def overlay_mask(image_path, mask_path, output_path, opacity=0.6):
    # Load the base image and mask
    base_img = Image.open(image_path).convert('RGBA')
    mask = Image.open(mask_path).convert('RGBA')

    # Resize mask to match base image size if needed
    if base_img.size != mask.size:
        mask = mask.resize(base_img.size, Image.LANCZOS)

    # Convert images to numpy arrays
    base_array = np.array(base_img)
    mask_array = np.array(mask)

    # Apply opacity to the mask
    mask_array[:, :, 3] = mask_array[:, :, 3] * opacity

    # Combine the base image and the mask
    combined = Image.alpha_composite(base_img, Image.fromarray(mask_array))

    # Save the result
    combined.save(output_path)
    print(f"Mask overlay image saved to {output_path}")

def save_ground_truth():
    for m in Path(args.results_dir).rglob("*/labels/*.png"):
        # Search the file with the same name in the results directory
        fp = [f for f in Path(args.output_dir).rglob(f"{m.name.split('mask_')[-1]}")]

        if fp:
            f = fp[0]
            # Extract the file name without the extension
            new_fp = f"{Path(args.output_dir).parent}/mask-overlay/{f.stem}.png"
            label = Image.open(m).convert('RGBA')
            lp = f"{Path(new_fp).parent.parent}/labels/{f.stem}.png"
            Path(lp).parent.mkdir(parents=True, exist_ok=True)

            label.save(lp)

            Path(new_fp).parent.mkdir(parents=True, exist_ok=True)
            overlay_mask(f, m, new_fp)

def save_results(regex = "mask_*.png"):
    for m in Path(args.results_dir).rglob(regex):
        # Search the file with the same name in the results directory
        fp = [f for f in Path(args.output_dir).rglob(f"{m.stem.split('mask_')[-1]}*")]

        if fp:
            f = fp[0]

            # Extract the file name without the extension
            new_fp = f"{Path(args.results_dir)}/output-overlay/{f.stem}.png"
            Path(new_fp).parent.mkdir(parents=True, exist_ok=True)
            overlay_mask(f, m, new_fp)


def save_gt_images():
    for datalist in [train_ds, val_ds, test_ds]:
        dataloader = DataLoader(datalist, batch_size=1, shuffle=False)

        for idx, (batch, path) in tqdm(enumerate(dataloader), total=len(dataloader)):

            if isinstance(batch, dict):
                batch = {
                    x: batch[x].to(device)
                    for x in batch if isinstance(batch[x], torch.Tensor)
                }
            elif isinstance(batch, list):
                batch = [x.to(device) for x in batch if isinstance(x, torch.Tensor)]
            elif isinstance(batch, torch.Tensor):
                batch = batch.to(device)
            else:
                print("not support data type")
            current_img_np = batch["image"].squeeze().cpu().numpy()
            current_img_np = (current_img_np - current_img_np.min()) / (current_img_np.max() - current_img_np.min())
            current_img_np = (current_img_np * 255).astype(np.uint8)
            image_array = cv2.cvtColor(current_img_np, cv2.COLOR_GRAY2BGR)
            # save the image as a png
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, f'{path[0].split(".")[0]}.png'), image_array)


def save_gt_masks(save_dir = 'visualization'):

    for datalist in [train_ds, val_ds, test_ds]:
        dataloader = DataLoader(datalist, batch_size=1, shuffle=False)

        for idx, (batch, path) in tqdm(enumerate(dataloader), total=len(dataloader)):

            if isinstance(batch, dict):
                batch = {
                    x: batch[x].to(device)
                    for x in batch if isinstance(batch[x], torch.Tensor)
                }
            elif isinstance(batch, list):
                batch = [x.to(device) for x in batch if isinstance(x, torch.Tensor)]
            elif isinstance(batch, torch.Tensor):
                batch = batch.to(device)
            else:
                print("not support data type")
            output_mask = batch["label"].squeeze().cpu().numpy()
            color_mask = np.zeros((320, 320, 3), dtype=np.uint8)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            for i in range(3):
                mask = output_mask[i]
                colored_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2) * colors[i]
                color_mask = cv2.add(color_mask, colored_mask.astype(np.uint8))


            # save the image as a png
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, f'{path[0].split(".")[0]}.png'), color_mask)

def load_images_to_rgb(red_path, green_path, blue_path):
    """
    Load three separate grayscale images and combine them into a single RGB image.

    Parameters:
    red_path (str): Path to the image for the red channel
    green_path (str): Path to the image for the green channel
    blue_path (str): Path to the image for the blue channel

    Returns:
    PIL.Image: Combined RGB image
    """
    # Load each channel as a grayscale image
    red_channel = Image.open(red_path).convert('L')
    green_channel = Image.open(green_path).convert('L')
    blue_channel = Image.open(blue_path).convert('L')

    # Ensure all images have the same size
    if red_channel.size != green_channel.size or red_channel.size != blue_channel.size:
        raise ValueError("All input images must have the same dimensions")

    # Convert PIL images to numpy arrays
    red_array = np.array(red_channel)
    green_array = np.array(green_channel)
    blue_array = np.array(blue_channel)

    # Stack the arrays to create an RGB image
    rgb_array = np.stack((red_array, green_array, blue_array), axis=2)

    # Convert the numpy array back to a PIL Image
    rgb_image = Image.fromarray(rgb_array.astype('uint8'), 'RGB')

    return rgb_image


def combine_channels_to_rgb(directory='results', channels_regex=["2/", "1/", "0/"]):
    # Iterate over files in the 0, 1, 2 subfolders
    for filename in Path(directory).rglob(f'*{channels_regex[0]}*'):
        if not str(filename.name).startswith('original_'):
            # Construct paths for each channel
            red_path = filename #os.path.join(directory,  channels_regex[0], filename)
            green_path = str(filename).replace(channels_regex[0], channels_regex[1])#os.path.join(directory, channels_regex[1], filename)
            blue_path = str(filename).replace(channels_regex[0], channels_regex[2])

            # Check if all channel files exist
            if os.path.exists(red_path) and os.path.exists(green_path) and os.path.exists(blue_path):
                # Open each channel image
                red_channel = Image.open(red_path).convert('L')
                green_channel = Image.open(green_path).convert('L')
                blue_channel = Image.open(blue_path).convert('L')

                # Combine channels into RGB image
                rgb_image = Image.merge('RGB', (red_channel, green_channel, blue_channel))

                # Save the combined RGB image
                output_filename = f"masks/combined_mask_{str(filename.name.replace(channels_regex[0],'')).rsplit('.', 1)[0]}.png"
                output_path = os.path.join(directory, output_filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                rgb_image.save(output_path)
                print(f"Saved combined image: {output_path}")
            else:
                print(f"Skipping {filename}: Not all channel files found")



def save_channelwise_mask(channels_regex = ["*0/", "*1/", "*2/"], regex = "[!original]_*.png"):

    for m in Path(args.results_dir).rglob(f"{channels_regex[0]}{regex}"):

        # Search the file with the same name in the results directory
        fp = [f for f in Path(args.output_dir).rglob(f"{m.stem.split('mask_')[-1]}*")]

        if fp:
            f = fp[0]

            # Extract the file name without the extension
            new_fp = f"{Path(args.results_dir)}/output-overlay/{f.stem}.png"
            Path(new_fp).parent.mkdir(parents=True, exist_ok=True)
            overlay_mask(f, m, new_fp)

# Example usage:
# rgb_image = load_images_to_rgb('red_channel.png', 'green_channel.png', 'blue_channel.png')
# rgb_image.save('combined_rgb_image.png')

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/SPIDER_T1wT2w", type=str)
parser.add_argument("--fold", default=0, type=int)
parser.add_argument("--num_classes", default=4, type=int)
parser.add_argument("--output_dir", default="data/visualization/SPIDER_T1wT2w/images", type=str)
parser.add_argument("--results_dir", default="results/IISDM/SPIDER_T1wT2w/", type=str)
parser.add_argument("--regex", default="*masks/combined_*.png", type=str)
args = parser.parse_args()

data_dir = args.data_dir
split_path = os.path.join(data_dir, "splits_final.json")
output_dir = args.output_dir
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds, val_ds, test_ds = get_dataloader_SPIDER(data_dir=Path(data_dir),
                                                  split_path=split_path,
                                                  num_classes=3,
                                                  fold=0,
                                                  semantic_segmentation=True,
                                                  presegmentation=False,
                                                  train_val_test=False)


# save_gt_images()
#
# save_gt_masks(save_dir=Path(output_dir).parent/'labels')


# overlay the masks
#combine_channels_to_rgb(directory=args.results_dir, channels_regex=["output_2000/2/", "output_2000/1/", "output_2000/0/"])
#save_results(args.regex)

#find . -type f -path '*/output-overlay/*' -name "*SPIDER_010*" -exec bash -c 'mkdir -p visualization/$(dirname "{}") && cp "{}" "visualization/{}"' \;
vis_path = Path('results/visualization/SPIDER_009')
print( [i for i in sorted(vis_path.rglob('*/SPIDER_T1w/*.png')) if '_t1_' not in i.name])
images = [Image.open(i.as_posix()) for i in sorted(vis_path.rglob('*SPIDER_T1w/*/*.png'))]
images.extend([Image.open(i.as_posix()) for i in sorted(vis_path.rglob('*SPIDER_T2w/*/*.png'))])
images.extend([Image.open(i.as_posix()) for i in sorted(vis_path.rglob('*SPIDER_T1wT2w/*/*.png')) if '_t1_' not in i.name])

make_grid(images, rows=3, cols = 6).save(vis_path /'results_overview.png')