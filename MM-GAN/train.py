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
# import pandas as p
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
root = 'E:/gzj/MM-GAN/ckpt'
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
    epoch_sum = 200
    for epoch in range(epoch_sum):
        D_losses = []
        D_real_losses = []
        D_fake_losses = []
        G_train_l1_losses = []
        G_train_losses = []
        G_losses = []
        synth_losses = []

        # patient: Whole patient dictionary containing image, seg, name etc.
        # x_patient: Just the images of a single patient
        # x_r: Batch of images taken from x_patient according to the batch size specified.
        # x_z: Batch from x_r where some sequences are imputed with noise for input to G
        epoch_start_time = time.time()
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:#################################x修改的库pillow10.0.0---5.3
            #pip install scikit-image==0.15.0 -U -i https://pypi.tuna.tsinghua.edu.cn/simple 从0.21
            for images in tqdmDataLoader:
                # Put the whole patient in GPU to aid quicker training
                xray = images[0].to(device).float()
                xray = xray.unsqueeze(1)

                ct = images[1].to(device).float()
                ct = ct.unsqueeze(1)

                # test = images[0]
                # test = np.squeeze(test)
                # save_sample_ct = sitk.GetImageFromArray(test.cpu().numpy())
                # sitk.WriteImage(save_sample_ct,
                #                 os.path.join('E:/gzj/MM-GAN/ckpt/tensor1_input.nii.gz'))
                # test = images[1]
                # test = np.squeeze(test)
                # save_sample_ct = sitk.GetImageFromArray(test.cpu().numpy())
                # sitk.WriteImage(save_sample_ct,
                #                 os.path.join('E:/gzj/MM-GAN/ckpt/tensor2_input.nii.gz'))

                # create batches out of this patient

                x_r = torch.cat([xray, ct],dim=1)

                rand_val = torch.randint(low=0, high=len(scenarios), size=(1,))

                label_scenario = scenarios[int(rand_val.numpy()[0])]

                print('\tTraining this batch with Scenario: {}'.format(label_scenario))

                # create a new x_imputed and x_real with this label scenario
                x_z = x_r.clone().cuda()
                label_list_r = torch.from_numpy(np.ones((opt.batch_size,
                                                         patch[0],
                                                         patch[1],
                                                         patch[2],
                                                         patch[3],
                                                         ))).cuda().type(torch.cuda.FloatTensor)

                impute_tensor = torch.zeros((opt.batch_size,
                                                 opt.img_height,
                                                 opt.img_width,
                                                 opt.img_depth
                                                 ), device=device)

                for idx, k in enumerate(label_scenario):
                    if k == 0:
                        x_z[:, idx, ...] = impute_tensor
                        # print(label_list)
                        label_list[:, idx] = 0
                        # print(label_list)

                    elif k == 1:
                        label_list[:, idx] = 1

                # TRAIN GENERATOR G
                print('\tTraining Generator')
                generator.zero_grad()
                optimizer_G.zero_grad()

                fake_x = generator(x_z)

                pred_fake = discriminator(fake_x, x_r)

                if pred_fake.size() != label_list_r.size():
                    print('Error!')
                    import sys
                    sys.exit(-1)

                loss_GAN = criterion_GAN(pred_fake, label_list_r)

                # pixel-wise loss
                loss_pixel = criterion_pixelwise(fake_x, x_r)
                synth_loss = mse_fake_vs_real(fake_x, x_r)

                # variable that sets the relative importance to loss_GAN and loss_pixel
                lam = 0.9
                G_train_total_loss = (1 - lam) * loss_GAN + lam * loss_pixel

                G_train_total_loss.backward()
                optimizer_G.step()

                # save the losses
                G_train_l1_losses.append(loss_pixel.item())
                G_train_losses.append(loss_GAN.item())
                G_losses.append(G_train_total_loss.item())
                synth_losses.append(synth_loss.item())

                # TRAIN DISCRIMINATOR D
                # this takes in the real x as X-INPUT and real x as Y-INPUT
                print('\tTraining Discriminator')
                discriminator.zero_grad()
                optimizer_D.zero_grad()

                # real loss
                # EDIT: We removed noise addition
                # We can add noise to the inputs of the discriminator
                pred_real = discriminator(x_r,
                                          x_r)

                loss_real = criterion_GAN(pred_real, label_list_r)

                # fake loss
                fake_x = generator(x_z)

                # tag1
                # if opt.ic == 1:
                #     fake_x = impute_reals_into_fake(x_z, fake_x, label_scenario)

                # we add noise to the inputs of the discriminator here as well
                pred_fake = discriminator(fake_x.detach(), x_r)
                # pred_fake = discriminator(fake_x, x_r)

                loss_fake = criterion_GAN(pred_fake, label_list)

                D_train_loss = 0.5 * (loss_real + loss_fake)

                # for printing purposes
                D_real_losses.append(loss_real.item())
                D_fake_losses.append(loss_fake.item())
                D_losses.append(D_train_loss.item())

                D_train_loss.backward()
                optimizer_D.step()



                logger.info('loss_d: [real: %.5f, fake: %.5f, comb: %.5f], loss_g: [gan: %.5f, l1: %.5f, comb: %.5f], synth_loss_mse(ut): %.5f' % (
                                torch.mean(torch.FloatTensor(D_real_losses)),
                                torch.mean(torch.FloatTensor(D_fake_losses)),
                                torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_train_losses)),
                                torch.mean(torch.FloatTensor(G_train_l1_losses)), torch.mean(torch.FloatTensor(G_losses)),
                            torch.mean(torch.FloatTensor(synth_losses))))
                # Check if we have trained with exactly opt.train_patient_idx patients (if opt.train_patient_idx is 10, then idx_pat will be 9, so this condition will evaluate to true


        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print(
            '[%d/%d] - ptime: %.2f, loss_d: [real: %.5f, fake: %.5f, comb: %.5f], loss_g: [gan: %.5f, l1: %.5f, comb: %.5f], '
            'synth_loss_mse(ut): %.5f' % (
            (epoch + 1), opt.n_epochs, per_epoch_ptime, torch.mean(torch.FloatTensor(D_real_losses)),
            torch.mean(torch.FloatTensor(D_fake_losses)),
            torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_train_losses)),
            torch.mean(torch.FloatTensor(G_train_l1_losses)), torch.mean(torch.FloatTensor(G_losses)),
            torch.mean(torch.FloatTensor(synth_losses))))

        # Checkpoint the models

        gen_state_checkpoint = {
                'epoch': epoch + 1,
                'arch': opt.model_name,
                'state_dict': generator.state_dict(),
                'optimizer' : optimizer_G.state_dict(),
            }

        des_state_checkpoint = {
            'epoch': epoch + 1,
            'arch': opt.model_name,
            'state_dict': discriminator.state_dict(),
            'optimizer': optimizer_D.state_dict(),
        }

        torch.save(generator.state_dict(), os.path.join("E:/gzj/MM-GAN/ckpt/Gen.pt"))


        save_checkpoint(gen_state_checkpoint, os.path.join(root, 'generator_param_{}_{}.pkl'.format(model, epoch + 1)),
                        pickle_module=pickle)

        save_checkpoint(des_state_checkpoint,
                        os.path.join(root, 'discriminator_param_{}_{}.pkl'.format(model, epoch + 1)),
                        pickle_module=pickle)


        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

        # train_hist['test_loss']['mse'].append(result_dict_test['mean']['mse'])
        # train_hist['test_loss']['psnr'].append(result_dict_test['mean']['psnr'])
        # train_hist['test_loss']['ssim'].append(result_dict_test['mean']['ssim'])

        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

        end_time = time.time()
        total_ptime = end_time - start_time
        train_hist['total_ptime'].append(total_ptime)

        print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
        torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), opt.n_epochs, total_ptime))

if __name__ == '__main__':
    main()
