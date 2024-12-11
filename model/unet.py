"""
This experiment aims to explore which part of the skip connections in Unet is more important.\n
Using ASPP
"""
from typing import Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base import BaseModel
from model.modules import CA, CAB, CBA, CBR, CR, SE, CloseGate, DepthWiseConv2d, GroupConv2d, OpenCloseGate, OpenGate, PNNorm, PixelNorm, QKPlus, SEPlus, ShuffleUp
from model.utils import interlaced_cat, pad2same
import einops as eop



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, groups=1, with_bn=True, activate=nn.ReLU()):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels * groups
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels) if with_bn else nn.Identity(),
            activate,
            GroupConv2d(mid_channels, out_channels, 3, 1, 1, bias=False, out_groups=groups),
            nn.BatchNorm2d(out_channels * groups) if with_bn else nn.Identity(),
            activate
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, with_bn=True, groups=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, with_bn=with_bn, groups=groups)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, mode='cat', with_bn=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, with_bn=with_bn)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // (2 if mode=='cat' else 1), kernel_size=3, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, with_bn=with_bn)
        self.mode = mode

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1) if self.mode == 'cat' else x1 + x2
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x).sigmoid()

class UNet(BaseModel):
    def __init__(self, n_channels, n_classes, bilinear=False, return_feature=False, bottle=nn.Identity()):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        factor = 2 if bilinear else 1
        self.down4 = (Down(256, 512 // factor))
        self.up1 = (Up(512, 256 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up3 = (Up(128, 64 // factor, bilinear))
        self.up4 = (Up(64, 32, bilinear))
        self.outc = (OutConv(32, n_classes))
        self.return_feature = return_feature
        self.bottle = bottle

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.bottle(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits if not self.return_feature else (logits, x5)
    
    @staticmethod
    def create():
        return UNet(3, 1)
