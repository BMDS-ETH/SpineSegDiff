from pathlib import Path
from typing import Dict, Tuple
import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from torch import nn as nn
from src.guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from src.guided_diffusion.resample import UniformSampler
from src.guided_diffusion.respace import SpacedDiffusion, space_timesteps
from src.networks import BasicUNetEncoder, DenoisingUNet
from src.training.lr_scheduler import LinearWarmupCosineAnnealingLR
from src.training.trainer import Trainer
from src.training.utils import save_new_model_weights
from src.results.metric import dice


def uncertainty_weights(pred_out: torch.Tensor) -> torch.Tensor:
    """Compute the uncertainty of the prediction as the negative log of the prediction
    \begin{equation}
        \text{uncertainty} = - \log(p)
    \end{equation}

    Parameters:
    ----------
    pred_out: torch.tensor
        The output of the model
    Returns:
    -------
    uncer_out: torch.tensor
        The uncertainty of the model
    """
    pred_out = torch.sigmoid(pred_out)
    pred_out[pred_out < 0.001] = 0.001
    uncer_out = - pred_out * torch.log(pred_out)
    return uncer_out



class SpineSegDiff(nn.Module):  # nn.Module is the base class for all neural network modules in pytorch

    """ Class for the guided diffusion model for spine segmentation.

    Parameters
    ----------
    num_timesteps: int
        The number of timesteps
    presegmentation: bool
        Whether to use the pretrained model
    modality: int
        The number of modalities
    num_targets: int
        The number of targets
    skip_timesteps: int
        The number of timesteps to skip
    device: str
        The device to use
    weights_path: str
        The path to the weights
    """

    def __init__(self, num_timesteps,
                 presegmentation = False,
                 modality = 1,
                 num_targets = 3,
                 skip_timesteps = 50,
                 device = "cuda:0",
                 weights_path = "./weights"):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_classes = num_targets
        self.presegmentation = presegmentation
        self.features_sizes = [64, 64, 128, 256, 512, 64]
        self.image_size = (320, 320)
        self.device = device
        self.embed_model = BasicUNetEncoder(2, modality, self.num_classes , self.features_sizes)
        self.weights_path = weights_path
        self.model = DenoisingUNet(2, modality + self.num_classes , self.num_classes ,self.features_sizes,
                                   act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))

        betas = get_named_beta_schedule("cosine", self.num_timesteps)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(self.num_timesteps, [self.num_timesteps]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )

        if self.presegmentation:
            self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(self.num_timesteps, [self.num_timesteps]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        else:
            self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(self.num_timesteps, [skip_timesteps]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        self.sampler = UniformSampler(self.num_timesteps)


    def forward(self, image=None, x=None, pred_type=None, step=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            #get a timestep t and the probabity of this timestep sampler.sample calculates this probability distribution and samples from it
            t, weight = self.sampler.sample(x.shape[0], x.device)

            #now we sample the noisey image and retrun noisy image, timestep t and the noise that was added
            #in q_sample
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            #gets all embeddings of each convolutional layer block in total we have 5 embedding layers [x0,x1,x2,x3,x4]
            embeddings = self.embed_model(image)
            #train a u net, give the embeddings as input and the noised mask plus the corresponding image to it
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)

            if not self.presegmentation:
                sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1,  self.num_classes, self.image_size[0], self.image_size[1]), model_kwargs={"image": image, "embeddings": embeddings})
            else:
                sample_out = self.sample_diffusion.ddim_sample_loop_presegmentation(self.model,
                                                                                    (1,  self.num_classes, self.image_size[0], self.image_size[1]), x_pre = x, t_max=self.num_timesteps, model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out





class SpineSegDiffInference(SpineSegDiff):
    def __init__(self, num_timesteps,
                 presegmentation=False,
                 modality=1,
                 num_targets=3,
                 skip_timesteps=50,
                 device="cuda:0",
                 weights_path="../weights",
                 uncertainity_step = 10,
                 Ts = 15):

        super(  ).__init__( num_timesteps, presegmentation, modality , num_targets, skip_timesteps, device, weights_path)
        self.uncertainity_step = uncertainity_step
        self.Ts = Ts

    def forward(self, image=None, x=None, pred_type=None, step=None, embedding=None):

        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image) if embedding is None else embedding
            return self.model(x, t=step, image=image, embedding=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)

            sample_outputs = []
            for i in range(self.uncertainity_step):
                if self.presegmentation:
                    sample_out = self.sample_diffusion.ddim_sample_loop_presegmentation(self.model, (1,  self.num_classes, self.image_size[0], self.image_size[1]), x_pre = x, t_max=self.stratified_timesteps, model_kwargs={"image": image, "embeddings": embeddings})
                    sample_outputs.append(sample_out)
                else:
                    sample_outputs.append(self.sample_diffusion.ddim_sample_loop(self.model, (1,  self.num_classes, self.image_size[0], self.image_size[1]), model_kwargs={"image": image, "embeddings": embeddings}))

            sample_return = torch.zeros((1,  self.num_classes, self.image_size[0], self.image_size[1] ))
            sample_return = sample_return.to(self.device)
            total_uncertainy = []

            for index in range(self.Ts):
                uncer_out = 0
                for i in range(self.uncertainity_step):
                    uncer_out += sample_outputs[i]["all_model_outputs"][index]
                uncer_out = uncer_out / self.uncertainity_step
                uncer = uncertainty_weights(uncer_out).to(self.device)
                total_uncertainy.append(uncer)


                w = torch.exp(torch.sigmoid(torch.tensor((index + 1) / self.Ts)) * (1 - uncer)).to(self.device)


                for i in range(self.uncertainity_step):
                    sample = sample_outputs[i]["all_samples"][index]
                    sample_on_device = sample.to(self.device)
                    weighted_sample = w * sample_on_device
                    sample_return += weighted_sample

            return sample_return, total_uncertainy



class SpineSegDiffTrainer(Trainer):
    def __init__(self, env_type: str,
                 max_epochs: int,
                 batch_size: int,
                 device: str = "cpu",
                 val_every: int = 1,
                 num_gpus: int = 1,
                 logdir: str = "./logs/",
                 weights_path: str = "",
                 num_timesteps: int = 1000,
                 presegmentation: bool = False):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir)

        self.num_timesteps = num_timesteps
        self.presegmentation = presegmentation
        self.weights_path = weights_path if weights_path else logdir + f"/weights/" # weights_path
        self.model = SpineSegDiff(self.num_timesteps,
                                  self.presegmentation,
                                  modality=1,
                                  num_targets=3,
                                  skip_timesteps=50,
                                  device=device,
                                  weights_path= self.weights_path
                                  )

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-3)
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                       warmup_epochs=100,
                                                       max_epochs=max_epochs)

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)
        self.validation_metric = DiceMetric(include_background=False, reduction="mean")
        self.segmentations_labels = {
            "BG": 0, # Background
            "SC": 1, # Spinal Canal
            "VB": 2, # Vertebral Body
            "IVD": 3 # InterVertebral Disc
        }


    def training_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float]:
        # here we define the training step that is performed at each iteration! we get the batch of data here our diffusion process takes place!

        if not self.presegmentation:
            image, label = self.get_input(batch)
            x_start = label
        else:
            image, label, label_pre = self.get_input(batch)
            x_start = label_pre

        # we take our label mask, from batch["label"] that has shape (1,depth,height,widht) i guess

        x_start = (x_start) * 2 - 1  # why are we doing this exactly when labels are only 0 or 1 ??
        # we rerange the labels: either -1 or 1 for the mask values.
        # this is where the sampling
        # 1, we sample with UniformSampler(1000), this initializes the weights
        # 2 we use the sample function of the schedulesampler abc class: we get the probability distribution with probability = weight / sum(weights)
        # since all weight are the same --> all probabilites are the same
        # 3 we select a random timepoint: with indices_np = np.random.choice(len(p), size=(batch_size,), p=p) according to our probability distribution p
        # 4 we use the function q_sample. Here a lot happens
        # 4.1 we first calculate our beta schedule. this is how much variance noise we add for each timestep t,
        # 4.2 we calulculate the sqrt_alphas_cumprod multiply it with x_start
        # add sqrt_one_minus_alphas_cumprod and multiply it with the noise
        # now we have our q_sample
        # with shape: batchsize, channels_labels, depth, height, width
        # 5 we have our x_t, t and noise
        # shapes are x_t: batchsize, channels_labels, depth, height, width, noise: batchsize, channels_labels, depth, height, width, t: indiceies for each image in the batch batchsize

        x_t, t, noise = self.model(x=x_start, pred_type="q_sample")

        # we denoise the label, here we pass the noised label in, the image, and the timesteps t
        # 1. use a Unet Encoder to get the embeddings
        # U net encoder has spatial dims = 3, in_channels = number_modalities, out_channels = number_targets, features = [64, 64, 128, 256, 512, 64]
        # we get embeddings with
        # 2. use a U net that accepts this embeddings; inputs are images and noised image xt

        # U net decoder has spatial dims = 3, in_channels = number_modalities + number targets, out_channels = number_targets, features = [64, 64, 128, 256, 512, 64]
        # we image will be concatenated with the noised mask x_t in dim = 1
        # we caluclate the time embeddings and also input them to the U-net in the decoder and encoder

        pred_xstart = self.model(x=x_t, step=t, image=image, pred_type="denoise")

        # we predict various losses between our denoised label and the original label
        loss_dice = self.dice_loss(pred_xstart, label)
        loss_bce = self.bce(pred_xstart, label)

        pred_xstart = torch.sigmoid(pred_xstart)
        loss_mse = self.mse(pred_xstart, label)

        loss = loss_dice + loss_bce + loss_mse

        # calculate the lr and optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # calculate the lr
        lr = self.optimizer.param_groups[0]["lr"]

        return loss, lr

    def get_input(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = batch["image"]
        image = image.float()

        if self.presegmentation:
            label = batch["label"]
            label_pre = batch["label_pre"]
            label = label.float()
            label_pre = label_pre.float()
            return image, label, label_pre
        else:
            label = batch["label"]
            label = label.float()
            return image, label

    def validation_step(self, batch: Dict[str, torch.Tensor], path=None):
        """Perform the validation step.
                 Parameters:
                 ------------
                 batch: dict
                     The batch of data.
                 Returns:
                 ---------
                 dice_scores: list
                     The dice scores for each label.
                 output: np.array
                     The predicted output.
                 0: int
                     uncertainty score but isn't used in validation during training"""

        with torch.no_grad():
            if not self.presegmentation:
                image, label = self.get_input(batch)
                output = self.model(x=label, step=self.num_timesteps, image=image, pred_type="ddim_sample")
            else:
                image, label, label_pre = self.get_input(batch)
                output = self.model(x=label_pre, step=self.num_timesteps, image=image, pred_type="ddim_sample")
            output = torch.sigmoid(output)

            output = (output > 0.5).float().cpu().numpy()

            dice_scores = list()
            target = label.cpu().numpy()
            for i in range(output.shape[1]):
                o = output[:, i]
                t = target[:, i]
                dice_scores.append(dice(o, t))
            return dice_scores, output, 0

    def validation_end(self, mean_val_outputs):
        """Perform the validation end step by saving the best performing model.
        Parameters:
        ------------
        mean_val_outputs: list
            The mean validation outputs.
        Returns:
        ---------
        None
        """

        print("mean_val_outputs", mean_val_outputs)

        if len(mean_val_outputs) == 3:
            sc, ver, ivd = mean_val_outputs

            print("Spinal canal ", sc)
            print("vertebrae", ver)
            print("ivd's", ivd)
            print("mean_dice", (sc + ver + ivd) / 3)

            mean_dice = (sc + ver + ivd) / 3
        else:

            mean_dice = 0
            mean_val_outputs = [tensor.item() for tensor in mean_val_outputs]
            for i, val in enumerate(mean_val_outputs):
                print(f"Dice Score of Label {self.labels[i]}: ", val, step=self.epoch)
                mean_dice += val
            mean_dice /= len(mean_val_outputs)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_weights(self.model,
                                   (Path(self.weights_path) / f"best_model_{mean_dice:.4f}.pt").as_posix(),
                                    delete_regex="best_model")


        save_new_model_weights(self.model,
                               (Path(self.weights_path) / f"final_model_{mean_dice:.4f}.pt").as_posix(),
                               delete_regex="final_model")