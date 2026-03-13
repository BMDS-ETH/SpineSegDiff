import numpy as np
import torch
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.evaluation.metric import dice, recall, fscore
import argparse
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
from dataset.spider_data_utils_multi_label import get_loader_spider
from pathlib import Path
from monai.metrics import compute_hausdorff_distance
set_determinism(123)
import os
import matplotlib.pyplot as plt

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--data_dir", type=str, default="./datasets/Dataset005_SPIDER")
    parser.add_argument("-s","--semantic_segmentation", type=bool,default=True)
    parser.add_argument("-c","--num_classes", type=int,default=4)
    parser.add_argument("-l","--logdir", type=str,default="./logs_spider_patientwise_semantic_seg_both/diffusion_seg_all_loss_embed/model")
    parser.add_argument("-f","--fold", type=int,default=0)
    parser.add_argument("-e","--num_ensemble", type=int,default=1)
    args = parser.parse_args()
    return args

args = arg_parser()

#data_dir = "./datasets/Dataset007_SPIDER_t2"

data_dir = args.data_dir
print(data_dir)

max_epoch = 100
batch_size = 4
val_every = 10
device = "cuda:0"
fold = args.fold
print(f"fold: {fold}")
number_modality = 1
number_targets = args.num_classes-1
print("number_targets",number_targets)
print("number of ensemble",args.num_ensemble)


def compute_uncer(pred_out):
    pred_out = torch.sigmoid(pred_out)
    pred_out[pred_out < 0.001] = 0.001
    uncer_out = - pred_out * torch.log(pred_out)
    return uncer_out

class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, number_modality, number_targets, [64, 64, 128, 256, 512, 64])

        self.model = BasicUNetDe(3, number_modality + number_targets, number_targets, [64, 64, 128, 256, 512, 64], 
                                act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))
        
        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [50]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        self.sampler = UniformSampler(1000)

    def forward(self, image=None, x=None, pred_type=None, step=None, embedding=None):

        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embedding=embedding)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)

            uncer_step = 4
            sample_outputs = []
            for i in range(uncer_step):
                sample_outputs.append(self.sample_diffusion.ddim_sample_loop(self.model, (1, number_targets, 32, 128, 128), model_kwargs={"image": image, "embeddings": embeddings}))

            sample_return = torch.zeros((1, number_targets, 32, 128, 128))
            sample_return = sample_return.to("cuda:0")

            for index in range(10):

                uncer_out = 0
                for i in range(uncer_step):
                    uncer_out += sample_outputs[i]["all_model_outputs"][index]
                uncer_out = uncer_out / uncer_step
                uncer = compute_uncer(uncer_out).to("cuda:0")

                w = torch.exp(torch.sigmoid(torch.tensor((index + 1) / 10)) * (1 - uncer)).to("cuda:0")

              
                for i in range(uncer_step):
                    sample = sample_outputs[i]["all_samples"][index]
                    sample_on_device = sample.to("cuda:0")
                    weighted_sample = w * sample_on_device
                    sample_return += weighted_sample

            return sample_return

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cuda:0", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py", ensemble = 1 ):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[32, 128, 128],
                                        sw_batch_size=1,
                                        overlap=0.5)
        
        self.model = DiffUNet()
        self.num_ensemble = ensemble



    def get_uncertainty_map(self,ensemble_output_sigmoid):

            num_channels = ensemble_output_sigmoid[0].shape[0]

            uncertainty_per_channel = []
            for channel in range(num_channels):
                channel_output = []
                for i in range(len(ensemble_output_sigmoid)):
                    channel_output.append(ensemble_output_sigmoid[i][channel,:,:])
                channel_output = torch.stack(channel_output)
                channel_output = torch.std(channel_output, dim=0)

                uncertainty_per_channel.append(channel_output)
                print(channel_output.shape, "channel output shape")
            return torch.stack(uncertainty_per_channel)
        


    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"]
       
        label = label.float()
        image = image.float()
        return image, label 


    def validation_step(self, batch):
        image, label = self.get_input(batch)
        image = image.to("cuda:0")
        label = label.to("cuda:0")
        self.model = self.model.to("cuda:0")
        print(label.shape, "label shape")

        with torch.no_grad():
            ensemble_output = []
            ensemble_output_sigmoid = []
            if self.num_ensemble > 1: # if we want to use ensemble and create uncertainty maps
                for i in range(self.num_ensemble):
                    output = self.window_infer(image, self.model, pred_type="ddim_sample")
                    print(output.shape, "output shape")



                    ensemble_output_sigmoid.append(torch.sigmoid(output[0,:,output.shape[2]//2,:,:])) # getting the ensemble output with sigmoid
                    ensemble_output.append(output.cpu().numpy()) # getting the ensemble output
                ensemble_output = [torch.tensor(arr) for arr in ensemble_output]
                output = torch.mean(torch.stack(ensemble_output), dim=0) # mean of the ensemble output for dice score¨
                print(output.shape, "output shape after ensembling")
                uncertainty = self.get_uncertainty_map(ensemble_output_sigmoid) # getting the uncertainty map¨

                print(uncertainty.shape, "uncertainty shape")


                print(output.shape, "with ensemble")
            else:
                for i in range(self.num_ensemble):
                    output = self.window_infer(image, self.model, pred_type="ddim_sample")
                uncertainty = 0
                print(output.shape)

        output = torch.sigmoid(output)

        output = (output > 0.5).float().cpu().numpy()
        output = output.astype(int)
        target = label.cpu().numpy()
        dice_scores = list()

        for i in range(output.shape[1]):
            
            o = output[:, i]
            t = target[:, i]

            assert len(np.unique(0)) <=2 and len(np.unique(t))<=2
            dice_scores.append(dice(o, t))
        print(f"dice_scores: {dice_scores}")
        return dice_scores, output, uncertainty
    


if __name__ == "__main__":

    split_path = os.path.join(data_dir,"splits_final.json")

    args = arg_parser()
    print(args.data_dir)
    data_dir = args.data_dir
    print(args.semantic_segmentation)
    semantic_segmentation = args.semantic_segmentation
    print(args.num_classes)
    num_classes = args.num_classes

    logdir = args.logdir
    train_ds, val_ds, test_ds = get_loader_spider(data_dir=Path(data_dir), split_path=split_path,num_classes=num_classes, batch_size=batch_size, fold = args.fold, semantic_segmentation=semantic_segmentation)
    
    trainer = BraTSTrainer(env_type="pytorch",
                                    max_epochs=max_epoch,
                                    batch_size=batch_size,
                                    device=device,
                                    val_every=val_every,
                                    num_gpus=1,
                                    master_port=17751,
                                    training_script=__file__,
                                    ensemble=args.num_ensemble)

    logdir = args.logdir
    models = os.listdir(os.path.join(logdir,"model"))
    best_model = 0
    for model_n in models:
        if model_n.startswith("best"):
            best_model = model_n
    if best_model ==0:
        best_model = [model for model in models if model.startswith("final")]
        if best_model == None:
            print(f"no model found in folder {logdir} check the train.py file")
    model_dir = os.path.join(logdir,"model",best_model)
    print(f'the best model was: {best_model}')

    trainer.load_state_dict(model_dir)
    v_mean, _ = trainer.validation_single_gpu(val_dataset=test_ds, save_path = args.logdir)

    print(f"v_mean is {v_mean}")