"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import nibabel as nib
import sys
import random
sys.path.append(".")
import numpy as np
import time
import torch as th
import torch.distributed as dist
from guided_diffusion.spiderloader import get_loader_spider

from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    train_ds, val_ds, test_ds = get_loader_spider(args.data_dir, 
                                                args.split_path, 
                                                args.num_classes,
                                                args.semantic_segmentation, 
                                                args.image_dimension,
                                                args.fold)
    
    datal = th.utils.data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False)
    

    data = iter(datal)
    all_images = []
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    while len(all_images) * args.batch_size < args.num_samples:
        b,_, path = next(data)  #should return an image from the dataloader "data"
        c = th.randn_like(_)
        img = th.cat((b, c), dim=1)     #add a noise channel$
    

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)


        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

            s = th.tensor(sample)
            #create a nifty image
            s = np.asarray(s.cpu().detach().numpy())
            s = np.where(s > 0, 1, 0)

            segmentation = np.argmax(s, axis=1)
            score = dice_score(segmentation, org)
            print(score, "DICE Score")
            print(segmentation.shape)
            print(path)
            segmentation_nifty = nib.Nifti1Image(segmentation[0,:,:], np.eye(4))
            save_path = os.path.join('./results', path, str(i) + '.nii.gz')



def create_argparser():
    data_dir="../data/SPIDER_T1wT2w"
    defaults = dict(
        data_dir="../data/SPIDER_T1wT2w",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="./results/savedmodel005000.pt",
        split_path = os.path.join(data_dir,"splits_final.json"),
        num_classes = 4, # we use semantic segmentation #TODO has to be specified in the dataset 4 for semantic
        semantic_segmentation = True,
        image_dimension = 2,
        mode = "test",
        fold = 0,
        num_ensemble=5
             #number of samples in the ensemble
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
