#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:04:44 2019

@author: tao
"""

import os
import torch
import torch.nn as nn
import numpy as np

import time
import datetime

from torch.utils.data import DataLoader

#from models import *
#from fusion_models import *  # revise in 09/03/2019
from dataset import MultiModalityData_load
from funcs.utils import *
import torch.nn as nn
import scipy.io as scio
from torch.autograd import Variable
import torch.autograd as autograd
#import IVD_Net as IVD_Net
import model.syn_model as models
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



#from config import opt
#from visualize import Visualizer
#testing     

#os.environ["CUDA_VISIBLE_DEVICES"] = '5,6'

cuda = True if torch.cuda.is_available() else False
FloatTensor   = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor    = torch.cuda.LongTensor if cuda else torch.LongTensor


class LatentSynthModel():
    
    ########################################################################### scikit-image0.21------0.16
    
    def __init__(self,opt):
        
        self.opt         = opt  
        self.generator   = models.Multi_modal_generator(1,1,32)
        self.discrimator = models.Discriminator()
        
        if opt.use_gpu: 
            self.generator    = self.generator.cuda()
            self.discrimator  = self.discrimator.cuda()
                        
        if torch.cuda.device_count() > 1:
            self.generator    = nn.DataParallel(self.generator,device_ids=self.opt.gpu_id)
            self.discrimator  = nn.DataParallel(self.discrimator,device_ids=self.opt.gpu_id)  
        


    ########################################################################### 
    def train(self):
        def read_image_xray(file_path):
            image = Image.open(file_path).convert("L")
            image = image.resize((256, 256))
            image = np.array(image)
            image = skimage.exposure.rescale_intensity(image, out_range=(-1, 1))  # -1,1
            imageLR = image
            return imageLR

        def read_image_ct(file_path):
            w = 2200
            l = -100
            image = sitk.ReadImage(file_path)
            image = sitk.GetArrayFromImage(image)
            lr_shape = (128, 128, 128)
            imageHR = scipy.ndimage.zoom(image, np.array(lr_shape) / np.array(image.shape), order=3)
            imageHR = skimage.exposure.rescale_intensity(imageHR, in_range=(l - w / 2, l + w / 2),
                                                         out_range=(-1, 1))  # -1,1
            return imageHR

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

        if not os.path.isdir(self.opt.save_path+'/'+'task_'+str(self.opt.task_id)+'/'):
            mkdir_p(self.opt.save_path+'/'+'task_'+str(self.opt.task_id)+'/')

        logger = Logger(os.path.join(self.opt.save_path+'/'+'task_'+str(self.opt.task_id)+'/'+'run_log.txt'), title='')
        logger.set_names(['Run epoch', 'D Loss', 'G Loss'])

        #
        self.generator.apply(weights_init_normal)
        self.discrimator.apply(weights_init_normal)
        print('weights_init_normal')
                
        # Optimizers
        optimizer_D     = torch.optim.Adam(self.discrimator.parameters(), lr=self.opt.lr,betas=(self.opt.b1, self.opt.b2))
        optimizer_G     = torch.optim.Adam(self.generator.parameters(),lr=self.opt.lr,betas=(self.opt.b1, self.opt.b2))

        # Learning rate update schedulers
        lr_scheduler_G  = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(self.opt.epochs, 0, self.opt.decay_epoch).step)
        lr_scheduler_D  = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(self.opt.epochs, 0, self.opt.decay_epoch).step)
    
            
        # Lossesgenerator
        criterion_GAN   = nn.MSELoss().cuda()
        criterion_identity = nn.L1Loss().cuda()

        # Load data
        device = torch.device("cuda:0")

        json_file_path = 'D:/Code/MICCAI2023/data.json'
        dataset = CustomDataset(
            json_file=json_file_path,
        )

        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(total_num)
            print(trainable_num)

        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=True, pin_memory=False, num_workers=1)  # shuffle??????????????

        batches_done = 0
        prev_time    = time.time()
        epoch_sum = 200
        # ---------------------------- *training * ---------------------------------
        for epoch in range(epoch_sum):
            with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
                for images in tqdmDataLoader:
                    #用两个生成1个
                    # define diferent synthesis tasks
                    # [x1,x2,x3] = model_task(inputs,self.opt.task_id) # train different synthesis task
                    xray = images[0].to(device).float()
                    xray = xray.unsqueeze(1)

                    ct = images[1].to(device).float()
                    ct = ct.unsqueeze(1)

                    fake  = torch.zeros([xray.shape[1]*xray.shape[0], 1, 6, 6], requires_grad=False) #.cuda()
                    valid = torch.ones([xray.shape[1]*xray.shape[0], 1, 6, 6], requires_grad=False)#.cuda()

                    ###############################################################

                    x_fu = torch.cat([ct,ct],dim=1)

                    # ----------------------
                    #  Train generator
                    # ----------------------
                    optimizer_G.zero_grad()

                    x_fake,x1_re,x2_re = self.generator(x_fu)

                    # Identity loss
                    loss_re3 = criterion_identity(x_fake, xray)
                    loss_re1 = criterion_identity(x1_re, ct)
                    loss_re2 = criterion_identity(x2_re, ct)


                    # gan loss
                    loss_GAN = criterion_GAN(self.discrimator(x_fake), valid)

                    # total loss
                    loss_G = loss_GAN + 100*loss_re3 + 20*loss_re1 + 20*loss_re2


                    loss_G.backward(retain_graph=True)
                    optimizer_G.step()

                    # ----------------------
                    #  Train Discriminators
                    # ----------------------
                    optimizer_D.zero_grad()


                    # Real loss
                    loss_real = criterion_GAN(self.discrimator(xray), valid)
                    loss_fake = criterion_GAN(self.discrimator(x_fake), fake)
                    # Total loss
                    loss_D = (loss_real + loss_fake) / 2

                    loss_D.backward(retain_graph=True)
                    optimizer_D.step()

                    # time
                    tqdmDataLoader.set_postfix(ordered_dict={
                        "epoch": epoch,
                        "loss: ": loss_D.item(),
                    })

                    batches_done += 1


                # Update learning rates
                lr_scheduler_G.step()
                lr_scheduler_D.step()

                logger.append([epoch, loss_D.item(), loss_G.item()])

                # Save model checkpoints
                if epoch % 10 == 0 or epoch == 199:
                    torch.save(self.generator.state_dict(), os.path.join(
                        'D:/Code/Hi-Net/ckpt/', 'generator_ckpt_' + str(epoch) + "_.pt"))
                    torch.save(self.discrimator.state_dict(), os.path.join(
                        'D:/Code/Hi-Net/ckpt/', 'discrimator_ckpt_' + str(epoch) + "_.pt"))


    ###########################################################################
    def test(self,ind_epoch):   
         
        self.generator.load_state_dict(torch.load(self.opt.save_path+'/'+'task_'+str(self.opt.task_id)+'/'+ 'generator_'+str(ind_epoch)+'.pkl'),strict=False)   
       
        # Load data        
        te_data   = MultiModalityData_load(self.opt,train=False,test=True)
        te_loader = DataLoader(te_data,batch_size=self.opt.batch_size,shuffle=False)
        
        pred_eva_set = []
        for ii, inputs in enumerate(te_loader): 
            #print(ii) 
            # define diferent synthesis tasks
            [x_in1, x_in2, x_out] = model_task(inputs,self.opt.task_id)
            x_fusion   = torch.cat([x_in1,x_in2],dim=1)
                      
            if self.opt.use_gpu:
                x_fusion     = x_fusion.cuda()
            
            
            # pred_out -- [batch_size*4,1,128,128]
            # x3       -- [batch_size*4,1,128,128]
            pred_out,pred_out1,pred_out2 = self.generator(x_fusion) 


            errors = prediction_syn_results(pred_out,x_out)  
            
            print(errors)
           
            pred_eva_set.append([errors['MSE'],errors['SSIM'],errors['PSNR']])
        
        mean_values = [ind_epoch,np.array(pred_eva_set)[:,0].mean(),np.array(pred_eva_set)[:,1].mean(),np.array(pred_eva_set)[:,2].mean(),np.array(pred_eva_set)[:,3].mean(),np.array(pred_eva_set)[:,4].mean(),np.array(pred_eva_set)[:,5].mean()]

        return mean_values
    
    
    