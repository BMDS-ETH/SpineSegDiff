from pathlib import Path
import os
import argparse
import logging
from monai.utils import set_determinism
from src.dataset.DataloaderSPIDER import get_dataloader_SPIDER
from src.networks.LumbarSpineSegDiff import LumbarSpineSegDiffTrainer

set_determinism(123)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--data_dir", type=str, default="./data/Dataset_SPIDER_T2w")
    parser.add_argument("-b","--batch_size", type=int,default=4)
    parser.add_argument("-v","--val_every", type=int,default=50)
    parser.add_argument("-vs","--val_start", type=int,default=200)
    parser.add_argument("-c","--num_classes", type=int,default=4)
    parser.add_argument("-s","--semantic_segmentation", type=bool,default=True)
    parser.add_argument("-l","--logdir", type=str,default= "./results/SpinsegDiff-T2w/fold_0")
    parser.add_argument("-t","--timesteps", type=int,default= 1000)
    parser.add_argument("-f","--fold", type=int,default= 0)
    parser.add_argument("-p","--presegmentation", type=bool, default= False)
    parser.add_argument("-tr","--train_val_test", type=bool,default= False)
    parser.add_argument("-e","--epochs", type=int,default= 1500)


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()
    env = "pytorch"  # or env = "pytorch" if you only have one gpu. #ddp
    logdir = args.logdir
    weights_path = os.path.join(logdir, "weights")
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Example usage of logging
    logger = logging.getLogger(__name__)

    max_epoch = args.epochs
    batch_size = args.batch_size
    fold = args.fold
    data_dir = args.data_dir
    num_timesteps = args.timesteps
    val_every = args.val_every
    num_gpus = 1
    device = "cuda:0"
    number_modality = 1
    number_targets = args.num_classes - 1  ## all channels that we would need :)
    split_path = os.path.join(data_dir,"splits_final.json")
    batch_size = args.batch_size
    num_classes = args.num_classes
    num_timesteps = args.timesteps
    presegmentation = args.presegmentation
    presegmentation = False
    semantic_segmentation = args.semantic_segmentation
    logger.info(f"we use this dataset: {data_dir}")
    logger.info(f"Number of Classes  {num_classes}")
    logger.info(f"Number of Targets: {number_targets}")

    logger.info(f"Precondition:  {presegmentation}")
    train_val_test = args.train_val_test
    logger.info(f"Validation every : {val_every} epoch starting from {args.val_start}")

    train_ds, val_ds, test_ds = get_dataloader_SPIDER(data_dir=Path(data_dir),
                                                      split_path=split_path,
                                                      num_classes= num_classes,
                                                      fold = args.fold,
                                                      semantic_segmentation=semantic_segmentation,
                                                      presegmentation= presegmentation,

                                                      train_val_test=train_val_test)
    

    
    model_trainer = LumbarSpineSegDiffTrainer (env_type=env,
                                        max_epochs=max_epoch,
                                        batch_size=batch_size,
                                        device=device,
                                        logdir=logdir,
                                        weights_path=weights_path,
                                        val_every=val_every,
                                        num_gpus=num_gpus,
                                        num_timesteps =num_timesteps,
                                        presegmentation= presegmentation)

    model_trainer.train(train_dataset=train_ds, val_dataset=val_ds)

