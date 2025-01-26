import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
np.random.seed(1337)
torch.manual_seed(1337)
random.seed(1337)
torch.backends.cudnn.deterministic = True
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

##############################
#           U-NET
##############################

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv3d(in_size, out_size, 3, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm3d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [  #nn.ConvTranspose3d(in_size, out_size, 3, 2, 1, bias=False),
                    # nn.InstanceNorm3d(out_size),
                    nn.ReLU(inplace=False)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, with_tanh=False, with_relu=False):
        super(GeneratorUNet, self).__init__()

        # original dropout was 0.5
        self.down1 = UNetDown(in_channels, 16, normalize=False)
        self.down2 = UNetDown(16, 16, normalize=False)
        self.down3 = UNetDown(16, 32, normalize=False)
        self.down4 = UNetDown(32, 64, normalize=False,dropout=0.2)
        self.down5 = UNetDown(64, 64, normalize=False,dropout=0.2)

        self.up1 = UNetUp(128, 64, dropout=0.2)
        self.up2 = UNetUp(128, 64, dropout=0.2)
        self.up3 = UNetUp(96, 64, dropout=0.2)
        self.up4 = UNetUp(80, 64, dropout=0.2)


        self.up = nn.Upsample(scale_factor=2)

        if with_tanh:

            self.final = nn.Sequential(
                nn.Upsample(scale_factor=2),
                # nn.ConstantPad3d((1, 0, 1, 0, 1, 0), 0),
                nn.Conv3d(192, out_channels, 3, padding=1),
                nn.Tanh()
            )
        elif with_relu:
            # this is for ISLES2015
            self.final = nn.Sequential(
                nn.Upsample(scale_factor=2),
                # nn.ConstantPad3d((1,0,1,0,1,0),0),
                nn.Conv3d(192, out_channels, 3, padding=1),
                nn.ReLU()
            )

        else:
            self.final = nn.Sequential(
                nn.Upsample(scale_factor=2),
                # nn.ConstantPad3d((1,0,1,0,1,0),0),
                nn.Conv3d(192, out_channels, 3, padding=1)
            )


    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(self.up(d5), d4)
        u2 = self.up2(self.up(u1), d3)
        u3 = self.up3(self.up(u2), d2)
        u4 = self.up4(self.up(u3), d1)

        return self.final(u4)


##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, dataset='BRATS2018'):
        super(Discriminator, self).__init__()

        # inp, stride, pad, dil, kernel = (256, 2, 1, 1, 8)
        # np.floor(((inp + 2*pad - dil*(kernel - 1) - 1)/stride) + 1)

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv3d(in_filters, out_filters, 3, stride=1, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels*2, 16, normalization=False),
            *discriminator_block(16, 16),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            # nn.ConstantPad3d((1,0,1,0,1,0),0),
            nn.Conv3d(64, out_channels, 3, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
