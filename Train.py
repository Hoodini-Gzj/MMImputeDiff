
import os
from typing import Dict
import skimage.io
import skimage.exposure
import torch
torch.cuda.current_device()
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
# from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from Diffusiondir.Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer ,GaussianDiffusionSampler1
from Diffusiondir.Model import UNet
# from Diffusiondir.VIT import DiT
from Diffusiondir.UVIT import UViT
from Scheduler import GradualWarmupScheduler
import scipy
from torch.utils.data import Dataset
from PIL import Image
import os
import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import json
from monai.transforms import Compose, RandSpatialCrop, RandRotate90, AddChannel, RandFlip, RandScaleIntensity, RandShiftIntensity, RandGaussianSmooth, RandAffine, RandGaussianSharpen, ToTensor

import torch.nn.functional as F

def read_image_xray(file_path):
    image = Image.open(file_path).convert("L")
    image = np.array(image)
    image = skimage.exposure.rescale_intensity(image, out_range=(-1, 1))  # -1,1
    return image

def read_image_ct(file_path):
    w = 2600
    l = 300
    image = sitk.ReadImage(file_path)
    image = sitk.GetArrayFromImage(image)
    shape = (128, 128, 128)
    return image

class CustomDataset(Dataset):
    def __init__(self, json_file, transform=None, transform1=None):
        with open(json_file, 'r') as f:
            self.data_pairs = json.load(f)
        self.transform = transform
        self.transform1 = transform1
    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        pair = self.data_pairs[idx]

        file_path_1 = pair['path1']
        file_path_2 = pair['path2']

        xray = read_image_xray(os.path.join(file_path_2))
        ct = read_image_ct(os.path.join(file_path_1))

        # if self.transform:

        return xray, ct

def read_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    json_file_path = './dataset.json'
    dataset = CustomDataset(
        json_file = json_file_path,
        transform=transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,)),
        ]),
        transform1=transforms.Compose([
            # transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,), (0.5,)),
        ])

    )
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(total_num)
        print(trainable_num)

    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=False, pin_memory=False, num_workers=10)#shuffle??????????????

    net_model = UViT().to(device)

    get_parameter_number(net_model)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=10, eta_min=1e-6, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
    # start training
    for e in range(modelConfig["epoch"]):
        totol_loss = 0
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images in tqdmDataLoader:
                # train
                optimizer.zero_grad()

                xray = images[0].to(device).float()
                xray = xray.unsqueeze(1)

                ct = images[1].to(device).float()
                ct = ct.unsqueeze(1)

                loss = trainer(xray, ct).sum()


                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                totol_loss += loss.item()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "xray shape: ": xray.shape,
                    "ct shape: ": ct.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        print(totol_loss)
        if e % 10 == 0 or e == 199:
            torch.save(net_model.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))


def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UViT()
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()

        sample_ct = []
        sample_xray = []
        mse = []
        # sum_sample_ct = []
        sum_sample_ct = torch.zeros(size=[modelConfig["batch_size"], 1, 96, 96, 96], device=device)
        for i in range(10):
            sampler = GaussianDiffusionSampler1(
                model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

            image = Image.open('./sub-S04401-neg/sub-S04401_ses-E08746_run-1_bp-chest_vp-pa_cr.png').convert("L")
            image = np.array(image)
            image = skimage.exposure.rescale_intensity(image, out_range=(-1, 1))  # -1,1
            w = 2200
            l = -100
            ConditionImage = image
            ConditionImage = torch.tensor(ConditionImage)
            ConditionImage = ConditionImage.to(device).float()
            ConditionImage = ConditionImage.unsqueeze(0)
            ConditionImage = ConditionImage.unsqueeze(0)
            NoisyImage = torch.randn(
                size=[modelConfig["batch_size"], 1, 128, 128, 128], device=device)

            sampledImgs_xray, sampledImgs_ct = sampler(ConditionImage, NoisyImage)#(xray-2D, CT-3D)
            sampledImgs_xray = np.squeeze(sampledImgs_xray)
            sampledImgs_ct = np.squeeze(sampledImgs_ct)

            show_image(sampledImgs_ct, os.path.join('./sampledImgs_ct_' + str(i) + '.png'))
            show_image_xray(sampledImgs_xray, os.path.join('./sampledImgs_xray_' + str(i) + '.png'))
            save_sample_ct = sitk.GetImageFromArray(sampledImgs_ct.cpu().numpy())
            sitk.WriteImage(save_sample_ct, os.path.join('./sampledImgs_ct_' + str(i) + '.nii.gz'))

            sample_ct.append(sampledImgs_ct)
            sample_ct.append(show_image_xray)
