import random
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base import BaseModel, LossWrap
from model.modules import CA, CBR, CR, DepthWiseConv2d, ModuleStack, PixelNorm, PositionalEncoding, Residual, Sum, _pair, pixelnorm
from model.utils import pad2same
from einops import rearrange
from torchvision.utils import save_image
from utils.topbreak import branch_based_invasion, random_dilate, random_over_invasion, skeletonize_tensor as hard_skel

from utils.losses import cldice, connection_loss, dice_loss
    

def cross_contain_loss(gates: List[torch.Tensor], first=False) -> torch.Tensor:
    if len(gates) <= 1: return 0
    y = gates.pop(0)
    pred = gates[-1]
    pred = F.interpolate(pred, size=y.shape[-2:], mode='bilinear', align_corners=False)
    if first:
        loss = cldice(pred, y)
        return loss
    else:
        tar_p = pred[y >= 0.5]
        loss_p = F.binary_cross_entropy(tar_p, torch.ones_like(tar_p)) if tar_p.numel() > 0 else 0
        tar_n = y[pred < 0.5]
        loss_n = F.binary_cross_entropy(tar_n, torch.zeros_like(tar_n)) if tar_n.numel() > 0 else 0
        loss = (loss_p + loss_n) / 2
    return loss + cross_contain_loss(gates)

def gate_enhance_loss(gate: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
    tar = F.interpolate(tar, size=gate.shape[-2:], mode='nearest')
    loss_dice = dice_loss(gate, tar, from_logits=True)
    # loss_dice = dice_loss(gate, (gate >= 0.5).float(), from_logits=True)
    loss_bce = F.binary_cross_entropy(gate, tar)
    loss_con = connection_loss(gate)
    return loss_dice + loss_con + loss_bce

def scs_loss(gate: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
    tar = F.interpolate(tar, size=gate.shape[-2:], mode='nearest')
    loss_dice = dice_loss(gate, tar, from_logits=True)
    gate_p = gate[tar > 0]
    loss_bi = F.binary_cross_entropy(gate_p, torch.ones_like(gate_p)) + gate.mean()
    return loss_dice + loss_bi

class DiscLoss(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.aligner = TopAlign(1, hidden_dim)
        self.optim = torch.optim.Adam(self.aligner.parameters(), lr=1e-4)

    def update(self, pred, tar):
        self.optim.zero_grad()
        pred = pred.clone().detach()
        align_p = self.aligner(pred, tar)
        p = random.random()
        if p < 0.1:
            t_tar, state = branch_based_invasion(tar, erosion_radius=random.randint(10, 60), offset=random.randint(-10, 10))
            state = (1 - state).reshape(-1, 1, 1, 1)
        elif p < 0.2:
            t_tar, state = random_over_invasion(tar, mode='center' if random.random() < 0.5 else 'corner')
            state = (1 - state).reshape(-1, 1, 1, 1)
        else:
            t_tar = tar
            state = 1
        align_t = self.aligner(t_tar, tar)
        loss = F.binary_cross_entropy(align_p, torch.zeros_like(align_p)) + F.binary_cross_entropy(align_t, torch.ones_like(align_t) * state)
        loss.backward()
        self.optim.step()

    def __call__(self, pred, tar):
        tar = F.interpolate(tar, size=pred.shape[-2:], mode='nearest')
        align = self.aligner(pred, tar)
        return F.binary_cross_entropy(align, torch.ones_like(align))
    
class AlignLoss(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.aligner = nn.Sequential(
            AlignConv(1, hidden_dim),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        self.optim = torch.optim.Adam(self.aligner.parameters(), lr=1e-4)

    def forward(self, pred, tar):
        tar = F.interpolate(tar, size=pred.shape[-2:], mode='nearest')
        aligned = self.aligner(pred)
        return F.binary_cross_entropy_with_logits(aligned, tar)


class CustomLoss(LossWrap):
    def __init__(self, aligner=None):
        super(CustomLoss, self).__init__(None)
        self.aligner = aligner
    
    def __call__(self, x, optimer):
        def _closure(pred):
            loss = {}
            optimer.zero_grad()
            mask, gates, scs = pred['mask'], pred['gates'], pred['scs']

            loss_bce = F.binary_cross_entropy_with_logits(mask, x.float())
            loss_dice = dice_loss(mask, x.float())
            loss_out = loss_bce + loss_dice
            loss['bce'] = loss_bce.item()
            loss['dice'] = loss_dice.item()
            loss['out'] = loss_out.item()

            loss_contrastive = sum(gate_enhance_loss(gate, x.float()) for gate in gates) / (len(gates))
            loss['contrastive'] = loss_contrastive.item()

            loss_bi = gate_enhance_loss(scs, x.float())
            loss['bi'] = loss_bi.item()

            # self.aligner.update(gates[-1], x.float())
            # loss_align = sum(self.aligner(gate, x.float()) for gate in gates) / 1
            # loss['align'] = loss_align.item()

            loss_total = loss_out + loss_contrastive + loss_bi
            loss['total'] = loss_total.item()

            loss_total.backward()
            optimer.step()
            return loss
        return _closure

class Preview(nn.Module):
    def __init__(self, in_channels, depth=4):
        super(Preview, self).__init__()
        self.view = nn.Sequential(
            *[
                nn.Sequential(
                    CBR(in_channels, in_channels, 3, 1, 1, groups=in_channels),
                    nn.MaxPool2d(2) 
                ) for _ in range(depth)           
            ],
            nn.Upsample(scale_factor=2**depth)
        )

    def forward(self, x):
        return self.view(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, with_embed=False):
        super(EncoderBlock, self).__init__()
        self.trans = nn.Sequential(
            CBR(in_channels, out_channels, 3, 1, 1),
            nn.Sequential(
                # Residual(Preview(out_channels)),
                PositionalEncoding(out_channels, 1000)
            ) if with_embed else nn.Identity(),
            CR(out_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm2d(out_channels, affine=True)
        )
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
        )
    
    def forward(self, x):
        if isinstance(x, tuple):
            h = self.trans(x[0][0])
            return (self.down(h), x[0][1]), F.relu(h)
        else:
            h = self.trans(x)
            down = self.down(h)
            return (down, down), F.relu(h)
    
def same_conv(conv:nn.Conv2d, x, dilation):
    kernel_size = conv.kernel_size
    dilation = _pair(dilation)
    _reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
    for d, k, i in zip(dilation, kernel_size,
                    range(len(kernel_size) - 1, -1, -1)):
        total_padding = d * (k - 1)
        left_pad = total_padding // 2
        _reversed_padding_repeated_twice[2 * i] = left_pad
        _reversed_padding_repeated_twice[2 * i + 1] = (
            total_padding - left_pad)
    return F.conv2d(F.pad(x, _reversed_padding_repeated_twice), 
                    conv.weight, conv.bias, 1, _pair(0), dilation)
 
class ScaleIt(nn.Module):
    def __init__(self, in_channels):
        super(ScaleIt, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 1, groups=in_channels, bias=False)
    
    def forward(self, x):
        return self.conv(self.conv(x))
    
class Gating(nn.Sequential):
    def __init__(self, in_channels, activation=nn.Sigmoid()):
        super(Gating, self).__init__(
            nn.InstanceNorm2d(in_channels),
            ScaleIt(in_channels),
            Sum(1, True),
            activation
        )

class AlignConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=[1, 3, 5, 7], fuse=True):
        super(AlignConv, self).__init__()
        self.strides = strides
        self.fuse = fuse
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.NRs = nn.ModuleList(
            [nn.Sequential(
                    nn.BatchNorm2d(out_channels), 
                    nn.LeakyReLU(), 
                    CA(out_channels, out_channels * 2, 1, groups=out_channels, activation=nn.LeakyReLU()) if fuse else nn.Identity(), # inner-outer
            ) 
            for _ in range(len(strides))]
        )
        self.fuse_outer = CA(out_channels, out_channels, 1, activation=nn.LeakyReLU()) if self.fuse else nn.Identity()
        self.out = CA(out_channels, out_channels, 1, activation=nn.LeakyReLU())
    
    def forward(self, x):
        y = 0
        for stride, BR in zip(self.strides, self.NRs):
            y = y + BR(same_conv(self.conv, x, stride))
        if not self.fuse:
            return self.out(y)
        y = rearrange(y, 'b (t c) h w -> b t c h w', t=2)
        i, o = torch.chunk(y, 2, dim=1)
        f_i = i.squeeze(1)
        f_o = self.fuse_outer(o.squeeze(1))
        return self.out(f_i + f_o)

class TopAlign(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, depth=1, kernel_size=3, strides=[1, 3, 5, 7]):
        super(TopAlign, self).__init__()
        self.strides = strides
        self.depth = depth
        self.conv = AlignConv(in_channels + 1, out_channels, kernel_size, strides)
        self.out = nn.Sequential(
            CBR(out_channels, out_channels * 2, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),
            CR(out_channels * 2, out_channels, 1),
            CA(out_channels, 1, 1, activation=nn.Sigmoid()),
        )
    
    def forward(self, x, c):
        x = random_dilate(x, random.randint(0, 3))
        c = c
        x = torch.cat([x, c], dim=1)
        y = 0
        for _ in range(self.depth):
            y = y + self.conv(x)
            x = F.max_pool2d(x, 2)
        return self.out(y)    
    
class SigConv(nn.Module):
    def __init__(self, in_channels):
        super(SigConv, self).__init__()
        # self.conv = nn.Conv2d(in_channels, 1, 1, bias=False)

    def weight(self, x):
        weights = torch.ones((*x.shape[:2], 1, 1), device=x.device)
        weights[:, x.size(1) // 2:, ...] *= 0
        return weights
        # return torch.sigmoid(self.conv.weight)


class EHGate(nn.Module):
    def __init__(self, in_channels):
        super(EHGate, self).__init__()
        self.tool = SigConv(in_channels)
        self.easyGate = nn.Sequential(
            CR(in_channels, in_channels, 1),
            Gating(in_channels, activation=nn.Tanh())
        )
        self.posGate = nn.Sequential(
            CR(in_channels, in_channels, 1),
            Gating(in_channels, activation=nn.Tanh())
        )
        self.negGate = nn.Sequential(
            CR(in_channels, in_channels, 1),
            Gating(in_channels, activation=nn.Tanh())
        )


    def forward(self, x):
        easy = self.tool.weight(x)
        hard = 1 - easy
        g_easy = self.easyGate(x * easy)
        f_hard = hard * x
        p_hard = f_hard * g_easy
        n_hard = f_hard * (1 - g_easy)
        p_hard = (p_hard - F.adaptive_avg_pool2d(p_hard, 1))
        n_hard = (n_hard - F.adaptive_avg_pool2d(n_hard, 1))
        gate = g_easy * 2 + self.posGate(p_hard) + self.negGate(n_hard)
        return gate / 4

class ShiftSigmoid(nn.Module):
    def __init__(self, shift=-3):
        super(ShiftSigmoid, self).__init__()
        self.shift = shift

    def forward(self, x):
        return torch.sigmoid(x + self.shift)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, with_skip=True):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.trans = CBR(out_channels, out_channels, 3, 1, 1)
        self.aux = nn.Sequential(
            CBR(out_channels, out_channels, 3, 1, 1),
            CA(out_channels, 1, 1, activation=nn.Sigmoid()),
        )

        if with_skip:
            # self.gate = EHGate(out_channels)
            self.fuse = nn.Sequential(
                CBR(out_channels, out_channels // 2, 3, 1, 1),
                CBR(out_channels // 2, out_channels // 2, 3, 1, 1),
                nn.Conv2d(out_channels // 2, out_channels, 1),
            )
        else:
            self.gate = None
            self.fuse = None

        self.reweight = nn.Sequential(
            CBR(out_channels, out_channels // 2, 3, 1, 1),
            CA(out_channels // 2, out_channels, 1, activation=nn.Sigmoid())
        )

        self.gate_log = nn.Identity()

    def forward(self, x, h):
        h_up = self.upsample(h[0])  
        h_up = pad2same(h_up, x[1])
        trans = self.trans(h_up)
        if self.fuse:
            gate = 1#self.gate(trans) 
            x_filter = gate * x[1]
            fuse = self.fuse(h_up + x_filter)
        else:
            fuse = trans
        reweight = self.reweight(fuse)
        out = fuse * reweight
        gate_com = self.aux(out)
        return out, h[1] + [gate_com]

class SimGate(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(SimGate, self).__init__()
        self.trans = AlignConv(in_channels, out_channels, fuse=False)
        self.trans_score = CR(out_channels, 1, 1)
        self.trans_pos_q = nn.Sequential(
            CR(mid_channels, mid_channels * 2, 1),
            CR(mid_channels * 2, mid_channels, 1),
            PixelNorm()
        )
        self.trans_pos_k = nn.Sequential(
            CR(in_channels, mid_channels, 1),
            PixelNorm()
        )
        self.gate_sim = nn.Sequential(
            nn.InstanceNorm2d(1),
            CA(1, 1, 1, activation=nn.Sigmoid()),
        )
    
    def forward(self, x, x_l):
        x_ = self.trans(x)
        map_score = self.trans_score(x_)
        B, _, H, W = map_score.shape
        score_map = map_score.flatten(1).softmax(-1).unsqueeze(-1)
        map_pos_q = self.trans_pos_q(x_l)
        map_pos_k = self.trans_pos_k(x)
        map_rel = torch.einsum('bchw,bcij->bhwij', map_pos_k, map_pos_q).reshape(B, (H * W), -1)
        template = torch.sum(map_rel * score_map, dim=1, keepdim=True)
        rel_map = torch.einsum('bcl,bCl->bcC', template, map_rel).reshape(map_score.shape)
        sim = self.gate_sim(rel_map)
        return x_ * sim, sim
    
class Bottleneck(nn.Module):
    """
    Assume that the primary regions are already extracted, non-local network is used to in place of the full connection. 
    reducing the parameters while filling the gaps.
    """
    def __init__(self, in_channels, out_channels, mid_channels):
        super(Bottleneck, self).__init__()
        self.com = SimGate(in_channels, out_channels, mid_channels)
        self.aux = nn.Sequential(
            CBR(out_channels, out_channels // 16, 3, 1, 1),
            CA(out_channels // 16, 1, 1, activation=nn.Sigmoid())
        )
    def forward(self, x): 
        y, g = self.com(*x[0])
        return y, [self.aux(y)]

class TANet(BaseModel):
    def __init__(self, in_channels, num_classes, mid_channels=32, depth=4):
        super(TANet, self).__init__()
        self.depth = depth
        self.mid = mid_channels
        self.topalign = nn.Sequential(
            AlignConv(1, 32),
            CA(32, 1, 1, activation=nn.Sigmoid())
        )
        self.default_closure = CustomLoss(self.topalign)
        self.net = self.build_stack(in_channels, mid_channels, depth)
        self.out = nn.Sequential(
            CBR(mid_channels, mid_channels, 3, 1, 1),
            nn.Conv2d(mid_channels, num_classes, 1),
            nn.Sigmoid()
        )

    def build_stack(self, in_channels, mid_channels, depth):
        if depth <= 0:
            return Bottleneck(in_channels, mid_channels, self.mid)
        
        return ModuleStack(
            EncoderBlock(in_channels, mid_channels, with_embed=depth==self.depth),
            self.build_stack(mid_channels, mid_channels * 2, depth - 1),
            DecoderBlock(mid_channels * 2, mid_channels)
        )

    def forward(self, x):  
        h, gates = self.net(x)
        y = self.out(h)
        scs = gates[0]
        gates = [self.topalign(gate) for gate in gates[1:]]
        self.pre = {
            'mask': y,
            'gates': gates,
            'scs': scs
        }
        return y
    

