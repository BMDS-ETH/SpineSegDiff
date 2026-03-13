"""
Train a diffusion model on images.
"""
import sys
import argparse
sys.path.append("..")
sys.path.append(".")
import os
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.spiderloader import get_loader_spider
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.train_util import TrainLoop

def main():
    ## Get the arguments... these are a lot
    args = create_argparser().parse_args()
    data_dir = args.data_dir
    split_path = args.split_path
    num_classes = args.num_classes
    semantic_segmentation = args.semantic_segmentation
    image_dimension = args.image_dimension
    mode = args.mode


    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    logger.log(f"We predict: {args.num_classes-1} classes")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)

    logger.log("creating data loader...")
    #TODO: Implement here the transforms for the data
    #make one for training and one for validation ---> use a function get_dataset_spider


    train_ds, val_ds, test_ds = get_loader_spider(args.data_dir, 
                                                  args.split_path, 
                                                  args.num_classes,
                                                  args.semantic_segmentation, 
                                                  args.image_dimension,
                                                  args.fold)
    
    if mode == "train":
        ds = train_ds
    else:
        ds = test_ds
    
    
    datal= th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)

    val_data = th.utils.data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False)
    val_datal = val_data

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        val_dataloader=val_datal,
        image_size = args.image_size,
        num_classes = args.num_classes,
        semantic_segmentation = args.semantic_segmentation,
        image_dimension = args.image_dimension,
        fold = args.fold,
        num_ensemble= args.num_ensemble,
        p_sample_loop_known = diffusion.p_sample_loop_known,
        ddim_sample_loop_known = diffusion.ddim_sample_loop_known,
        use_ddim = args.use_ddim,
        save_dir = args.save_dir

    ).run_loop()


def create_argparser():
    data_dir = "./data/Dataset012_SPIDER"
    defaults = dict(
        data_dir="./data/Dataset012_SPIDER",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=10,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',#'"./results/pretrainedmodel.pt",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        split_path = os.path.join(data_dir,"splits_final.json"),
        num_classes = 4, # we use semantic segmentation #TODO has to be specified in the dataset 4 for semantic
        semantic_segmentation = True,
        image_dimension = 2,
        mode = "train",
        fold = 4,
        num_ensemble = 10,
        use_ddim = False,
        save_dir = "./results/t1_f4",
        )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
