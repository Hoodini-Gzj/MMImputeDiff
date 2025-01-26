import os
import argparse
from modules.advanced_gans.models import *
from torch.autograd import Variable
from modules.models import cPix2PixDiscriminator
import time
import itertools
import pickle, gc
from modules.helpers import (ToTensor,
                             torch,
                             show_intermediate_results_BRATS,
                             Resize,
                             create_dataloaders,
                             impute_reals_into_fake,
                             save_checkpoint,
                             load_checkpoint,
                             generate_training_strategy,
                             calculate_metrics,
                             printTable)
import logging
import numpy as np
import copy, sys

try:
    logger = logging.getLogger(__file__.split('/')[-1])
except:
    logger = logging.getLogger(__name__)

# Ignore warnings
import warnings
import os
import argparse
from modules.advanced_gans.models import *
from torch.autograd import Variable
from modules.models import cPix2PixDiscriminator
import time
import itertools
import pickle, gc
from modules.helpers import (ToTensor,
                             torch,
                             show_intermediate_results_BRATS,
                             Resize,
                             create_dataloaders,
                             impute_reals_into_fake,
                             save_checkpoint,
                             load_checkpoint,
                             generate_training_strategy,
                             calculate_metrics,
                             printTable)
import logging
import numpy as np
import copy, sys

import argparse
import os
from tqdm import tqdm
import time
import logging
import random
import shutil
import numpy as np
# import pandas as pd
import nibabel as nib
import skimage.transform as skTrans
from matplotlib import pyplot as plt
import SimpleITK as sitk
from PIL import Image
import skimage.io
import skimage.exposure
from torch.utils.data import Dataset
from PIL import Image
import os
import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import json
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr
from torch.profiler import profile, record_function, ProfilerActivity
import torch.optim as optim
# import torchio as tio
import scipy
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=3, help='number of epochs of training')
parser.add_argument('--dataset', type=str, default="BRATS2018", help='name of the dataset')
parser.add_argument('--grade', type=str, default="LGG", help='grade of tumor to train on')
parser.add_argument('--path_prefix', type=str, default="", help='path prefix to choose')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=128, help='size of image height')
parser.add_argument('--img_width', type=int, default=128, help='size of image width')
parser.add_argument('--img_depth', type=int, default=128, help='size of image width')
parser.add_argument('--channels', type=int, default=2, help='number of image channels')
parser.add_argument('--out_channels', type=int, default=2, help='number of output channels')
parser.add_argument('--sample_interval', type=int, default=500, help='interval between sampling of images from generators')
parser.add_argument('--train_patient_idx', type=int, default=3, help='number of patients to train with')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
parser.add_argument('--discrim_type', type=int, default=1, help='discriminator type to use, 0 for normal, 1 for PatchGAN')
parser.add_argument('--test_pats', type=int, default=1, help='number of test patients')
parser.add_argument('--model_name', type=str, default='ckpt', help='name of mode')
parser.add_argument('--log_level', type=str, default='info', help='logging level to choose')
parser.add_argument('--c_learning', type=int, default=1, help='whether  or not use curriculum learning framework')
parser.add_argument('--use_tanh', action='store_true', help='use tanh normalization throughout')
parser.add_argument('--z_type', type=str, default='noise', help='what type of imputation method to use')
parser.add_argument('--ic', type=int, default=1, help='whether to use implicit conditioning (1) or not (0)')

opt = parser.parse_args()
print(opt)
root = 'D:/Code/MM-GAN/ckpt'
if 'info' in opt.log_level:
    logging.basicConfig(level=logging.INFO)
elif 'debug' in opt.log_level:
    logging.basicConfig(level=logging.DEBUG)

# Create Training and Validation data loaders

# notice there's one less asa224 here
# parent_path = os.path.join(opt.path_prefix, 'scratch/asa224/Datasets/BRATS2018/HDF5_Datasets/')


which_normalization = 'tanh'


# Initialize Networks

cuda = True if torch.cuda.is_available() else False

# =============================================================================
# Loss functions
# =============================================================================
criterion_GAN = torch.nn.BCELoss() if opt.discrim_type == 0 else torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()
mse_fake_vs_real = torch.nn.MSELoss()
# =============================================================================

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (opt.out_channels, opt.img_height, opt.img_width, opt.img_depth)


generator = GeneratorUNet(in_channels=opt.channels, out_channels=opt.out_channels, with_relu=True, with_tanh=False)
discriminator = Discriminator(in_channels=opt.channels, dataset='BRATS2018')


model = opt.model_name

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Send everything to GPU
if cuda:
    generator = nn.DataParallel(generator.cuda())
    discriminator = nn.DataParallel(discriminator.cuda())
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()
    mse_fake_vs_real.cuda()

    # Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


#  Training

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []
train_hist['test_loss'] = {
    'mse': [],
    'psnr': [],
    'ssim': []
}
# Get the device we're working on.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create all scenrios: Total will 15, but remove 0000 and 1111
scenarios = list(map(list, itertools.product([0, 1], repeat=2)))


# This is for D (Changed below)

label_list = torch.from_numpy(np.ones((opt.batch_size,
                                       patch[0],
                                       patch[1],
                                       patch[2],
                                       patch[3]
                                       ))).cuda().type(torch.cuda.FloatTensor)

# remove the empty scenario and all available scenario
scenarios.remove([0, 0])
scenarios.remove([1, 1])

# sort the scenarios according to decreasing difficulty. Easy scenarios last, and difficult ones first.
scenarios.sort(key=lambda x: x.count(1))

logger.info("Starting Training")
start_time = time.time()
def show_image_xray_cpu(imgs, fname=None, cmap='gray', norm=False, vmin=0, vmax=1, transpose='z', origin='lower'):
    fig, axes = plt.subplots(1, 1, figsize=(16, 16))
    # axes = axes.flatten()
    axes.imshow(imgs, cmap=plt.get_cmap(cmap), aspect='equal', origin=origin)
    if fname:
        fig.savefig(fname)
        plt.close(fig)
    else:
        return fig
def show_image(imgs, fname=None, cmap='gray', norm=False, vmin=0, vmax=1, transpose='z', origin='lower'):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    imgs = imgs.detach().cpu().numpy()
    # imgs = imgs.detach().numpy()
    for i, ax in zip(range(0, imgs.shape[0], imgs.shape[0] // 16), axes):
        ax.imshow(imgs[i], cmap=plt.get_cmap(cmap), aspect='equal', origin=origin)
    if fname:
        fig.savefig(fname)
        plt.close(fig)
    else:
        return fig
def show_image_cpu(imgs, fname=None, cmap='gray', norm=False, vmin=0, vmax=1, transpose='z', origin='lower'):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    for i, ax in zip(range(0, imgs.shape[0], imgs.shape[0] // 16), axes):
        ax.imshow(imgs[i], cmap=plt.get_cmap(cmap), aspect='equal', origin=origin)
    if fname:
        fig.savefig(fname)
        plt.close(fig)
    else:
        return fig
def read_image_xray(file_path):
    image = sitk.ReadImage(file_path)
    image = sitk.GetArrayFromImage(image)
    image = skimage.exposure.rescale_intensity(image, out_range=(-1, 1))
    return image

def read_image_ct(file_path):
    image = sitk.ReadImage(file_path)
    image = sitk.GetArrayFromImage(image)
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
        tag = pair['label']
        if tag == 'positive':
            label = 1
        else:
            label = 0

        xray = read_image_xray(os.path.join(file_path_2))
        ct = read_image_ct(os.path.join(file_path_1))

        return xray, ct, label

def read_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def main():
    device = torch.device("cuda:0")

    json_file_path = 'E:/gzj/MM-GAN/dataset.json'
    dataset = CustomDataset(
        json_file = json_file_path,
    )
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(total_num)
        print(trainable_num)

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, pin_memory=False, num_workers=1)#shuffle??????????????
    epoch_sum = 1
    for epoch in range(epoch_sum):

        # patient: Whole patient dictionary containing image, seg, name etc.
        # x_patient: Just the images of a single patient
        # x_r: Batch of images taken from x_patient according to the batch size specified.
        # x_z: Batch from x_r where some sequences are imputed with noise for input to G
        epoch_start_time = time.time()
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:#################################x修改的库pillow10.0.0---5.3
            #pip install scikit-image==0.15.0 -U -i https://pypi.tuna.tsinghua.edu.cn/simple 从0.21
            for images in tqdmDataLoader:
                # Put the whole patient in GPU to aid quicker training
                # test = images[0]
                # test = np.squeeze(test)
                # save_sample_ct = sitk.GetImageFromArray(test.cpu().numpy())
                # sitk.WriteImage(save_sample_ct,
                #                 os.path.join('E:/\gzj/MM-GAN/ckpt/tensor1_input.nii.gz'))

                xray = images[0].to(device).float()
                xray = xray.unsqueeze(1)

                ct = images[1].to(device).float()
                ct = ct.unsqueeze(1)

                # create batches out of this patient
                impute_tensor = torch.zeros((opt.batch_size,
                                             1,
                                             opt.img_height,
                                             opt.img_width,
                                             opt.img_depth
                                             ), device=device)
                x_r = torch.cat([xray, impute_tensor],dim=1)

                # 用于加载.pkl文件的文件路径
                ckpt = torch.load('E:/gzj/MM-GAN/ckpt/Gen.pt', map_location=device)
                generator.load_state_dict(ckpt)

                # 在这里，你现在可以使用已加载的权重运行你的模型了
                # 例如，你可以进行模型测试
                generator.eval()
                fake_x = generator(x_r)

                tensor1, tensor2 = torch.chunk(fake_x, chunks=2, dim=1)
                tensor1 = np.squeeze(tensor1)
                tensor2 = np.squeeze(tensor2)


                # tensor1_sample = np.squeeze(tensor1)
                # tensor1_sample = sitk.GetImageFromArray(tensor1_sample.detach().cpu().numpy())
                # sitk.WriteImage(tensor1_sample,
                #                 os.path.join('E:/gzj/MM-GAN/ckpt/tensor1.nii.gz'))



                show_image(tensor1,
                               os.path.join('E:/gzj/MM-GAN/ckpt/tensor1.png'))
                show_image(tensor2,
                               os.path.join('E:/gzj/MM-GAN/ckpt/tensor2.png'))
                print("1")



if __name__ == '__main__':
    main()
