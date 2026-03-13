import numpy as np
from dataset.spider_data_utils_multi_label import get_loader_spider
import torch
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
from monai.losses.dice import DiceLoss
from pathlib import Path
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
set_determinism(123)
import os
os.chdir(Path(__file__).resolve().parent)
import argparse


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--data_dir", type=str, default="./datasets/Dataset005_SPIDER")
    parser.add_argument("-b","--batch_size", type=int,default=2)
    parser.add_argument("-v","--val_every", type=int,default=20)
    parser.add_argument("-c","--num_classes", type=int,default=14)
    parser.add_argument("-s","--semantic_segmentation", type=bool,default=True)
    parser.add_argument("-l","--logdir", type=str,default= "./results/patientwise_semantic_seg_both")
    parser.add_argument("-f","--fold", type=int,default= 0)


    args = parser.parse_args()
    return args




env = "pytorch" # or env = "pytorch" if you only have one gpu. #ddp
args= arg_parser()
print("dataset used: ", args.data_dir)
logdir = args.logdir
model_save_path = os.path.join(logdir, "model")
max_epoch = 700
batch_size = args.batch_size
fold = args.fold


val_every = args.val_every
num_gpus = 1
device = "cuda:0"



number_modality =  1
number_targets = args.num_classes - 1 ## all channels that we would need :) 
print(number_targets)

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


    def forward(self, image=None, x=None, pred_type=None, step=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            #get a timestep t and the probabity of this timestep sampler.sample calculates this probability distribution and samples from it
            t, weight = self.sampler.sample(x.shape[0], x.device)
            
            #now we sample the noisey image and retrun noisy image, timestep t and the noise that was added
            #in q_sample 
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            #gets all embedings of each convolutional layer block in total we have 5 embedding layers [x0,x1,x2,x3,x4]
            embeddings = self.embed_model(image)
            #train a u net, give the embeddings as input and the noised mask plus the corresponding image to it
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)

            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, number_targets, 32, 128, 128), model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out

class BraTSTrainer(Trainer): #Implemeents the custom training loop for the BraTS dataset
      #at __init__ we specify what is important like the model, the optimizer, the loss and the noise scheduler
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[32, 128, 128],
                                        sw_batch_size=1,
                                        overlap=0.25)
        self.model = DiffUNet()

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-3)
        self.ce = nn.CrossEntropyLoss() 
        self.mse = nn.MSELoss()
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                  warmup_epochs=30,
                                                  max_epochs=max_epochs)

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)

        self.labels = {
    0: "SC",
    1: "L5",
    2: "L4",
    3: "L3",
    4: "L2",
    5: "L1",
    6: "T12",
    7: "T11",
    8: "T10",
    9: "T9",
    10: "IVD_L5",
    11: "IVD_L5L4",
    12: "IVD_L4L3",
    13: "IVD_L3L2",
    14: "IVD_L2L1",
    15: "IVD_L1T12",
    16: "IVD_T12T11",
    17: "IVD_T11T10",
    18: "IVD_T10T9"
}

    
    #herer we define the training step that is performed at each iteration! we get the batch of data here our diffusion process takes place!
    def training_step(self, batch):
        image, label = self.get_input(batch)

        #we take our label mask, from batch["label"] that has shape (1,depth,height,widht) i guess
        x_start = label

        x_start = (x_start) * 2 - 1 #why are we doing this exactly when labels are only 0 or 1 ??
        #we rerange the labels: either -1 or 1 for the mask values.


        #this is where the sampling 
        #1, we sample with UniformSampler(1000), this initializes the weights
        #2 we use the sample function of the schedulesampler abc class: we get the probability distribution with probability = weight / sum(weights)
        #since all weight are the same --> all probabilites are the same
        #3 we select a random timepoint: with indices_np = np.random.choice(len(p), size=(batch_size,), p=p) according to our probability distribution p
        #4 we use the function q_sample. Here a lot happens
            #4.1 we first calculate our beta schedule. this is how much variance noise we add for each timestep t,
            #4.2 we calulculate the sqrt_alphas_cumprod multiply it with x_start
            #add sqrt_one_minus_alphas_cumprod and multiply it with the noise
            #now we have our q_sample
            #with shape: batchsize, channels_labels, depth, height, width
        #5 we have our x_t, t and noise
        #shapes are x_t: batchsize, channels_labels, depth, height, width, noise: batchsize, channels_labels, depth, height, width, t: indiceies for each image in the batch batchsize
 
        x_t, t, noise = self.model(x=x_start, pred_type="q_sample")

        #we denoise the label, here we pass the noised label in, the image, and the timesteps t
        #1. use a Unet Encoder to get the embeddings
            # U net encoder has spatial dims = 3, in_channels = number_modalities, out_channels = number_targets, features = [64, 64, 128, 256, 512, 64]
            # we get embeddings with 
        #2. use a U net that accepts this embeddings; inputs are images and noised image xt

            # U net decoder has spatial dims = 3, in_channels = number_modalities + number targets, out_channels = number_targets, features = [64, 64, 128, 256, 512, 64]
            # we image will be concatenated with the noised mask x_t in dim = 1
            # we caluclate the time embeddings and also input them to the U-net in the decoder and encoder
        pred_xstart = self.model(x=x_t, step=t, image=image, pred_type="denoise")


        #we predict various losses between our denoised label and the original label
        loss_dice = self.dice_loss(pred_xstart, label)
        loss_bce = self.bce(pred_xstart, label)

        pred_xstart = torch.sigmoid(pred_xstart)
        loss_mse = self.mse(pred_xstart, label)

        loss = loss_dice + loss_bce + loss_mse

        self.log("train_loss", loss, step=self.global_step)

        return loss 
 
    def get_input(self, batch):
        #now the image has shape [batch, modalities, depth, height, width]
        #now the label has shape [batch, channels, depth, height, width]

        #each channel in the label is a binary mask for the corresponding class

        image = batch["image"]
        label = batch["label"]
       
        image = image.float()
        label = label.float()
        return image, label 

    def validation_step(self, batch):
        image, label = self.get_input(batch)    
        

        #TODO: documentate this
        #here we use the sliding window inferer from Monai library: it's used for inference on large images by just taking a small patch at a time
        #the window inferr takes in additional arguments in __call__ function: these are the input: here the image, the network: here the self.model (diffUnet) and kwargs for the network: pred_type="ddim_sample"

        #lets look at pred_type="ddim_sample"
        #takes in the embeddings of the Unet encoder,
        #calls the diffusion.ddim_sample_loop(self.model, (1, number_targets, 96, 96, 96), model_kwargs={"image": image, "embeddings": embeddings}) defined in the guided_diffusion.py

        #1 we call ddim_sample_loop
            #for each sample we call the ddim_sample_loop_progressive function
                #img is our noise here, we list our indices that are the list(range(self.num_timesteps))[::-1] --> meaning we get all timesteps starting from [num_time_steps-1, num_time_steps-2, ... 0]
                #we loop trhough all timesteps
                    #with th.no_grad_():
                    #we calculate the out: witht the function ddim_sample()
                    #this sample x_t-1 from the model using DDIM
                    #we now use the model: BasicUnetDe and input it into the function p_mean_variances
                    #we input the noise (=img), our timestep t, the image and the embeddings (both with model_kwargs)
                    #we get out "out". it consists of the mean and variance of the model
                        #p_mean_variances
                        #returns this:
                            #mean: the model mean output
                            #variance: the model variance output
                            #log_variance: the model log variance output
                            #pred_xstart: the model prediction for x_0
                    #afterwards we calculate the eps: epsilon with another function
                        #_predict_eps_from_xstart with input noise, timestep, and our prediction pred_xstart
                    #we get alpha bar
                    #we get alpha bar_prev
                    #we calculate sigma
                    #we produce new noise that is similar to x
                #from the ddim_sample we get the sample and store it as img
            #this sample is actually the output of the yiel out in the ddim_sample_loop_progressive function
            #we also save all samples in the dict final´
        


        output = self.window_infer(image, self.model, pred_type="ddim_sample")

        output = torch.sigmoid(output)

        output = (output > 0.5).float().cpu().numpy()


        dice_scores =list()
        target = label.cpu().numpy()
        for i in range(output.shape[1]):
            o = output[:, i]
            t = target[:, i]
            dice_scores.append(dice(o, t))
        return dice_scores, output, 0#dirty fix since we are not interested in the uncertainty during training

    def validation_end(self, mean_val_outputs):

    
        if len(mean_val_outputs) == 3 :
            sc, ver,ivd = mean_val_outputs

            print("Spincal cord ", sc)
            print("vertebrae", ver)
            print("ivd's", ivd)
            print("mean_dice", (sc+ver+ivd)/3)

            mean_dice =( sc +ver +ivd )/3
        else:
            
            mean_dice = 0
            mean_val_outputs = [tensor.item() for tensor in mean_val_outputs]
            for i, val in enumerate(mean_val_outputs):
                print(f"Dice Score of Label{self.labels[i]}: ",val, step=self.epoch)
                mean_dice += val
            mean_dice /= len(mean_val_outputs)


        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"best_model_{mean_dice:.4f}.pt"), 
                                            delete_symbol="best_model")

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model_{mean_dice:.4f}.pt"), 
                                        delete_symbol="final_model")

        #print(f"SC, {sc} Vertebrae: {ver} IVD's {ivd}, mean_dice is {mean_dice}")

if __name__ == "__main__":
    args = arg_parser()
    data_dir = args.data_dir
    print(f"we use this dataset: {data_dir}")
    split_path = os.path.join(data_dir,"splits_final.json")
    batch_size = args.batch_size
    num_classes = args.num_classes
    print("Number of Classes", num_classes)
    semantic_segmentation = args.semantic_segmentation
    print("Semantic Segmentation", semantic_segmentation)

    train_ds, val_ds, test_ds = get_loader_spider(data_dir=Path(data_dir), split_path=split_path,num_classes= num_classes ,batch_size=batch_size, fold = args.fold, semantic_segmentation=semantic_segmentation)
    #train_ds, val_ds, test_ds = get_loader_brats(data_dir=Path(data_dir), batch_size=batch_size, fold = 0)

    
    trainer = BraTSTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir=logdir,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17751,
                            training_script=__file__)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
