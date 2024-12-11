import torch
import torch.nn as nn

from model.base import BaseModel, LossWrap, wrap_iou
from model.modules import CR, SE, ModuleStack
from model.utils import pad2same
from utils.losses import DiceLoss


class MsFE(nn.Module):
    def __init__(self, in_channels):
        super(MsFE, self).__init__()
        self.par1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Conv2d(in_channels, in_channels, 5, padding='same'),
        )
        self.par2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Conv2d(in_channels, in_channels, 3, padding='same'),
        )
        self.par3 = nn.Sequential(
            nn.MaxPool2d(3, 1, padding=1),
            nn.Conv2d(in_channels, in_channels, 1),
        )
        self.par4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
        )
        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1 = self.par1(x)
        x2 = self.par2(x)
        x3 = self.par3(x)
        x4 = self.par4(x)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.out(out)
        # ??? 
        # as illustrated in original paper, the output is out + x, but it seems to be wrong, so I change it to out * x, 
        # or maybe out * x + x is correct, since the context declares that this module is inspired by residual block, but I am not sure.
        return out * x 
    
class MsAF(nn.Module):
    def __init__(self, in_channels):
        super(MsAF, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels * 2, in_channels, 3, 2)
        self.se = SE(in_channels, 1)
        self.mag = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding='same'),
            nn.Sigmoid(),
        )
        self.out = nn.Conv2d(in_channels * 2, in_channels, 3, padding='same')

    def forward(self, e_x, d_x):
        e_x = self.se(e_x)
        d_x = self.up(d_x)
        d_x = pad2same(d_x, e_x)
        ax = torch.mean(d_x, dim=1, keepdim=True)
        mx = torch.max(d_x, dim=1, keepdim=True)[0]
        mag = self.mag(torch.cat([ax, mx], dim=1)) * d_x
        out = torch.cat([e_x, mag], dim=1)
        out = self.out(out)
        return out

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 3, 2)
        self.msaf = MsAF(out_channels)
        self.msfe = MsFE(out_channels)
        self.conv = nn.Sequential(
            CR(out_channels * 2, out_channels, 3, padding='same'),
            CR(out_channels, out_channels, 3, padding='same'),
        )

    def forward(self, e, d):
        e_x = self.msfe(e[1])
        d_x = self.msaf(e_x, d[0])
        d_h = self.up(d[0])
        d_h = pad2same(d_h, e_x)
        y = torch.cat([d_x, d_h], dim=1)
        y = self.conv(y)
        return y,

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            CR(in_channels, out_channels, 3, padding='same'),
            CR(out_channels, out_channels, 3, padding='same'),
        )
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        y = self.conv(x[0])
        return self.down(y), y

class Bottle(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottle, self).__init__()
        self.conv = nn.Sequential(
            CR(in_channels, out_channels, 3, padding='same'),
            CR(out_channels, out_channels, 3, padding='same'),
        )

    def forward(self, x):
        y = self.conv(x[0])
        return y,

class MsAFUNet(BaseModel):
    def __init__(self, in_channels, num_classes):
        super(MsAFUNet, self).__init__()
        self.net = self.build_stack(in_channels, 32, 4)
        self.out = nn.Conv2d(32, num_classes, 1)

    def build_stack(self, in_channels, mid_channels, depth):
        if depth <= 0:
            return Bottle(in_channels, mid_channels)
        
        return ModuleStack(
            Encoder(in_channels, mid_channels),
            self.build_stack(mid_channels, mid_channels * 2, depth - 1),
            Decoder(mid_channels * 2, mid_channels)
        )

    def forward(self, x):  
        y = self.out(self.net((x, ))[0])
        return y
    
    default_closure = LossWrap(
            {
                'bce':{
                    'loss':nn.BCEWithLogitsLoss(),
                    'args':{}
                },
                'dice':{
                    'loss':DiceLoss(),
                    'args':{}
                }
            }
        )
