# Adapted from
# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging
import os

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from monai.data import DataLoader
import cv2
import matplotlib.pyplot as plt
from monai.utils import set_determinism
from src.training.sampler import SequentialDistributedSampler, distributed_concat



class Trainer:
    def __init__(self, env_type,
                 max_epochs,
                 batch_size,
                 device="cpu",
                 val_every=1,
                 num_gpus=1,
                 logdir="./logs/",
                 ):
        assert env_type in ["pytorch", "gpu", "cuda"], f"not support this env_type: {env_type}"
        self.env_type = env_type
        self.val_every = val_every
        self.max_epochs = max_epochs
        self.ddp = False
        self.num_gpus = num_gpus
        self.device = device
        self.rank = 0
        self.local_rank = 0
        self.batch_size = batch_size
        self.not_call_launch = True
        self.logdir = logdir
        self.scheduler = None
        self.model = None
        self.model_img = None
        self.auto_optim = True
        self.scheduler_img = None


        torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True

        gpu_count = torch.cuda.device_count()
        if num_gpus > gpu_count:
            logging.error("gpu not available" )
            os._exit(0)


    def get_dataloader(self, dataset, shuffle=False, batch_size=1, train=True):
        """Get the dataloader for the dataset.
        Parameters:
        -----------
        dataset: Dataset
            the dataset to be used
        shuffle: bool
            whether to shuffle the dataset
        batch_size: int
            the batch size
        train: bool
            whether the dataset is for training
        Returns:
        --------
        DataLoader
            the dataloader for the dataset
        """
        
        if dataset is None :
            return None
        if self.env_type == 'pytorch':
            return DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=1)
        else :
            if not train:
                sampler = SequentialDistributedSampler(dataset, batch_size=batch_size)

            else :
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
            return DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=12, 
                                sampler=sampler, 
                                drop_last=False)


    def validation_single_gpu(self, val_dataset,save_path = None):
        """validation for single gpu model called in test.py
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

        if self.ddp:
            print(f"single gpu model not support the ddp")
            exit(0)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        self.model.to(self.device)
        val_outputs = []
        self.model.eval()
        for idx, (batch,path) in tqdm(enumerate(val_loader), total=len(val_loader)):
            if isinstance(batch, dict):
                batch = {
                    x: batch[x].to(self.device)
                    for x in batch if isinstance(batch[x], torch.Tensor)
                }
            elif isinstance(batch, list) :
                batch = [x.to(self.device) for x in batch if isinstance(x, torch.Tensor)]

            elif isinstance(batch, torch.Tensor):
                batch = batch.to(self.device)
            
            else :
                print("not support data type")
                exit(0)

            with torch.no_grad():
                val_out, output, uncertainty = self.validation_step(batch, path[0])
                assert val_out is not None
                #plot the output and uncertainty
                if save_path is not None:
                    os.makedirs(os.path.join(save_path,"uncertainty"), exist_ok=True)
                    saving_path = os.path.join(save_path,"uncertainty", path[0].split(".")[0])
                    if isinstance(uncertainty, torch.Tensor):
                        print(uncertainty.shape)
                        for channel in range(uncertainty.shape[1]):
                            img = uncertainty[0,channel,:,:]
                            img = img.cpu().numpy()
                            plt.imsave(saving_path + f"uncertainty_{channel}.png", img, cmap='jet')

                    os.makedirs(os.path.join(save_path,"output"), exist_ok=True)
                    saving_path = os.path.join(save_path,"output", path[0].split(".")[0])
                    for channel in range(output.shape[1]):
                        img = output[0,channel,:,:]
                        assert len(img.shape) == 2
                        plt.imsave(saving_path + f"output_{channel}.png", img , cmap='gray')
                var = uncertainty
                imgage_dir = save_path
                if len(var.shape) == 4:
                    variance_map = var.squeeze(0).cpu().numpy()
                else:
                    variance_map = var.cpu().numpy()
                variance_map_normalized = np.zeros_like(variance_map)
                combined_variance_map_max = np.mean(variance_map, axis=0)
                combined_variance_map_mean = np.mean(variance_map, axis=0)
                for i in range(3):
                    vmin, vmax = variance_map[i].min(), variance_map[i].max()
                    variance_map_normalized[i] = cv2.normalize(variance_map[i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)   

                variance_map_normalized = variance_map_normalized.astype(np.uint8)
                variance_map_normalized = np.transpose(variance_map_normalized, (1, 2, 0))

                combined_variance_dir = os.path.join(imgage_dir, "combined_variance_maps")
                os.makedirs(combined_variance_dir, exist_ok=True)  # Create directory if it doesn't exist

                plt.imsave(os.path.join(combined_variance_dir,f'combined_variance_map_coolwarm{path[0].split(".")[0]}.png'),combined_variance_map_max, cmap="coolwarm")
                plt.imsave(os.path.join(combined_variance_dir,f'combined_variance_map_viridis{path[0].split(".")[0]}.png'),combined_variance_map_mean, cmap="viridis")

                #save the variance map as a png
                variance_dir = os.path.join(imgage_dir, "variance_maps")
                os.makedirs(variance_dir, exist_ok=True)  # Create directory if it doesn't exist
                cv2.imwrite(os.path.join(variance_dir,f'variance_map_{path[0].split(".")[0]}.png'),variance_map_normalized)


                current_seg_np = output[0,:,:,:]
                current_img_np = batch["image"].squeeze()

                current_img_np = current_img_np.cpu().numpy()
                # normalize img
                current_img_np = (current_img_np - current_img_np.min()) / (current_img_np.max() - current_img_np.min())
                current_img_np = (current_img_np * 255).astype(np.uint8)
                current_img_np = cv2.cvtColor(current_img_np, cv2.COLOR_GRAY2BGR)

                combined_mask = np.zeros((320, 320, 3), dtype=np.uint8)

                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] 
                for i in range(3):
                    mask = current_seg_np[i]
                    colored_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2) * colors[i]
                    combined_mask = cv2.add(combined_mask, colored_mask.astype(np.uint8))

                #save the combined mask img as a png
                mask_dir = os.path.join(imgage_dir, "masks")
                os.makedirs(mask_dir, exist_ok=True)  # Create directory if it doesn't exist

                cv2.imwrite(os.path.join(mask_dir,f'combined_mask_{path[0].split(".")[0]}.png'),combined_mask)

                

                alpha = 0.75
                blended_image = cv2.addWeighted(current_img_np, 1, combined_mask, alpha, 0)


                #save the image as a png
                cv2.imwrite(os.path.join(mask_dir,f'blended_image_{path[0].split(".")[0]}.png'),blended_image)




            #create a csv file with the dics scores results for each individual image
            csv_path = os.path.join(save_path, "dice_semantic.csv")
            if not os.path.exists(csv_path):
                with open(csv_path, "w") as file:
                    file.write("image,SC,Vertebrae,IVD\n")
                    file.write(f"{path[0].split('.')[0]},{val_out[0]},{val_out[1]},{val_out[2]}\n")
            else:
                with open(csv_path, "a") as file:
                    file.write(f"{path[0].split('.')[0]},{val_out[0]},{val_out[1]},{val_out[2]}\n")

            return_list = False
            val_outputs.append(val_out)

        if isinstance(val_out, list) or isinstance(val_out, tuple):
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
            else :
                v_sum = v_sum / length             
        else :
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
                else :
                    v_sum[i] = v_sum[i] / length[i]
        return v_sum, val_outputs

    def train(self,
                train_dataset,
                optimizer=None,
                optimizer_img = None,
                model=None,
                model_img = None,
                val_dataset=None,
                scheduler=None,
                scheduler_img=None,
              ):


        if scheduler is not None:
            self.scheduler = scheduler

        if scheduler_img is not None:
            self.scheduler_img = scheduler_img


        set_determinism(1234 + self.local_rank)

        if self.model is not None:
            logging.info(f"Label Model parameter: {next(self.model.parameters()).sum()}")
            para = sum([np.prod(list(p.size())) for p in self.model.parameters()])
            if self.local_rank == 0:
                logging.info(f"model parameters is {para * 4 / 1000 / 1000}M")

        if self.model_img is not None:
            logging.info(f"image model parameter: {next(self.model_img.parameters()).sum()}")
            para = sum([np.prod(list(p.size())) for p in self.model_img.parameters()])
            if self.local_rank == 0:
                logging.info(f"image model parameters is {para * 4 / 1000 / 1000}M ")
        

        self.global_step = 0
        if self.env_type == "pytorch":
            if self.model is not None:
                self.model.to(self.device)
            if self.model_img is not None:
                self.model_img.to(self.device)
            os.makedirs(self.logdir, exist_ok=True)


        elif self.ddp:
            if self.local_rank == 0:
                os.makedirs(self.logdir, exist_ok=True)
            if self.model is not None:
                self.model.cuda(self.local_rank)

                # self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                    device_ids=[self.local_rank],
                                                                    output_device=self.local_rank,
                                                                    find_unused_parameters=True)
            if self.model_img is not None:
                self.model_img.cuda(self.local_rank)
                self.model_img = torch.nn.parallel.DistributedDataParallel(self.model_img,
                                                                    device_ids=[self.local_rank],
                                                                    output_device=self.local_rank,
                                                                    find_unused_parameters=True)
        
        else :
            logging.error("not support env_type")
            exit(0)
        

        #load in the data with the train_loader and val_loader
        train_loader = self.get_dataloader(train_dataset, shuffle=True, batch_size=self.batch_size)
        if val_dataset is not None:
            val_loader = self.get_dataloader(val_dataset, shuffle=False, batch_size=1, train=False)
        else :
            val_loader = None 
        
        #iterate over each epoch
        for epoch in range(0, self.max_epochs):
            self.epoch = epoch 

            #dont quite get this but its ok
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
                torch.distributed.barrier()
            self.train_epoch(
                            train_loader,
                            epoch,
                            )
            
            val_outputs = []
            if (epoch+1) % self.val_every == 0 and epoch >= 400\
                    and val_loader is not None :
                if self.model is not None:
                    self.model.eval()
                if self.model_img is not None:
                    self.model_img.eval()

                if self.ddp:
                    torch.distributed.barrier()
                for idx, (batch,path) in tqdm(enumerate(val_loader), total=len(val_loader)):
                    if isinstance(batch, dict):
                        batch = {
                            x: batch[x].to(self.device)
                            for x in batch if isinstance(batch[x], torch.Tensor)
                        }
                    elif isinstance(batch, list) :
                        batch = [x.to(self.device) for x in batch if isinstance(x, torch.Tensor)]

                    elif isinstance(batch, torch.Tensor):
                        batch = batch.to(self.device)
                    
                    else :
                        logging.error("not support data type")
                        exit(0)

                    with torch.no_grad():
                    
                        val_out, output, uncertainty = self.validation_step(batch, path[0])


                        #plot the output and uncertainty
                        saving_path = os.path.join(self.logdir, path[0])
                        if  not isinstance(uncertainty, int):

                            for channel in range(uncertainty.shape[0]):
                                img = uncertainty[channel,:,:]
                                img = img.cpu().numpy()
                                plt.imsave(saving_path + f"uncertainty_{channel}.png", img, cmap='viridis')
                        assert val_out is not None 

                    return_list = False
                    val_outputs.append(val_out)
                    if isinstance(val_out, list) or isinstance(val_out, tuple):
                        return_list = True

                if self.ddp:
                    val_outputs = torch.tensor(val_outputs).to(self.device)
                    torch.distributed.barrier()

                    
                    val_outputs = distributed_concat(val_outputs, num_total_examples=len(val_loader.sampler.dataset))
                else :
                    val_outputs = torch.tensor(np.array(val_outputs))

                if self.local_rank == 0:
                    if not return_list:
                        length = 0
                        v_sum = 0.0
                        for v in val_outputs:
                            if not torch.isnan(v):
                                v_sum += v
                                length += 1

                        if length == 0:
                            v_sum = 0
                        else :
                            v_sum = v_sum / length 

                        self.validation_end(mean_val_outputs=v_sum)
                    
                    else :
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
                            else :
                                v_sum[i] = v_sum[i] / length[i]
                        print(len(v_sum))

                        self.validation_end(mean_val_outputs=v_sum)


            if self.scheduler is not None:
                self.scheduler.step()
            if self.scheduler_img is not None:
                self.scheduler_img.step()

            if self.model is not None:
                self.model.train()
            if self.model_img is not None:
                self.model_img.train()


    def train_epoch(self, 
                    loader,
                    epoch,
                    ):
        if self.model is not None:
            self.model.train()
            logging.info("label model is training")

        if self.model_img is not None:
            self.model_img.train()
            logging.info("image model is training")


        if self.local_rank == 0:
            with tqdm(total=len(loader)) as t:

                for idx, (batch,path) in enumerate(loader):
                    self.global_step += 1
                    t.set_description('Epoch %i' % epoch)
                    if isinstance(batch, dict):
                        batch = {
                            x: batch[x].contiguous().to(self.device)
                            for x in batch if isinstance(batch[x], torch.Tensor)
                        }
                    elif isinstance(batch, list) :
                        batch = [x.to(self.device) for x in batch if isinstance(x, torch.Tensor)]

                    elif isinstance(batch, torch.Tensor):
                        batch = batch.to(self.device)
                    
                    else :
                        print("not support data type")
                        exit(0)
                    

                    
                    if self.model_img is not None and self.model_img is not None:
                        loss, loss_img , lr, lr_img = self.training_step(batch)
                        t.set_postfix(loss=loss.item(), loss_img = loss_img.item(), lr=lr,lr_img = lr_img)
                    else:
                        loss, lr = self.training_step(batch)

                        t.set_postfix(loss=loss.item(), lr=lr)

                    t.update(1)


    def training_step(self, batch):
        raise NotImplementedError
    
    def validation_step(self, batch, path):
        raise NotImplementedError

    def validation_end(self, mean_val_outputs):
        pass 

    def save_mask(self, output, saving_path):
        pass

    def save_uncertainty(self, uncertainty, saving_path):
        pass

    def load_state_dict(self, weight_path, strict=True):
        sd = torch.load(weight_path, map_location="cpu")
        if "module" in sd :
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k 
            new_sd[new_k] = v

        self.model.load_state_dict(new_sd, strict=strict)
        print(f"model parameters are loaded successed.")