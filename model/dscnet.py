# https://github.com/YaoleiQi/DSCNet/tree/main
# -*- coding: utf-8 -*-
import torch
from torch import nn, cat
from torch.nn.functional import dropout
import einops

from model.base import BaseModel, LossWrap

"""Dynamic Snake Convolution Network"""
class DSConv_pro(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
        morph: int = 0,
        if_offset: bool = True,
        device: str | torch.device = "cuda",
    ):
        """
        A Dynamic Snake Convolution Implementation

        Based on:

            TODO

        Args:
            in_ch: number of input channels. Defaults to 1.
            out_ch: number of output channels. Defaults to 1.
            kernel_size: the size of kernel. Defaults to 9.
            extend_scope: the range to expand. Defaults to 1 for this method.
            morph: the morphology of the convolution kernel is mainly divided into two types along the x-axis (0) and the y-axis (1) (see the paper for details).
            if_offset: whether deformation is required,  if it is False, it is the standard convolution kernel. Defaults to True.

        """

        super().__init__()

        if morph not in (0, 1):
            raise ValueError("morph should be 0 or 1.")

        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = torch.device(device)
        self.to(device)

        # self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.gn_offset = nn.GroupNorm(kernel_size, 2 * kernel_size)
        self.gn = nn.GroupNorm(out_channels // 4, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, 3, padding=1)

        self.dsc_conv_x = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

    def forward(self, input: torch.Tensor):
        # Predict offset map between [-1, 1]
        offset = self.offset_conv(input)
        # offset = self.bn(offset)
        offset = self.gn_offset(offset)
        offset = self.tanh(offset)

        # Run deformative conv
        y_coordinate_map, x_coordinate_map = get_coordinate_map_2D(
            offset=offset,
            morph=self.morph,
            extend_scope=self.extend_scope,
            device=self.device,
        )
        deformed_feature = get_interpolated_feature(
            input,
            y_coordinate_map,
            x_coordinate_map,
        )

        if self.morph == 0:
            output = self.dsc_conv_x(deformed_feature)
        elif self.morph == 1:
            output = self.dsc_conv_y(deformed_feature)

        # Groupnorm & ReLU
        output = self.gn(output)
        output = self.relu(output)

        return output


def get_coordinate_map_2D(
    offset: torch.Tensor,
    morph: int,
    extend_scope: float = 1.0,
    device: str | torch.device = "cuda",
):
    """Computing 2D coordinate map of DSCNet based on: TODO

    Args:
        offset: offset predict by network with shape [B, 2*K, W, H]. Here K refers to kernel size.
        morph: the morphology of the convolution kernel is mainly divided into two types along the x-axis (0) and the y-axis (1) (see the paper for details).
        extend_scope: the range to expand. Defaults to 1 for this method.
        device: location of data. Defaults to 'cuda'.

    Return:
        y_coordinate_map: coordinate map along y-axis with shape [B, K_H * H, K_W * W]
        x_coordinate_map: coordinate map along x-axis with shape [B, K_H * H, K_W * W]
    """

    if morph not in (0, 1):
        raise ValueError("morph should be 0 or 1.")

    batch_size, _, width, height = offset.shape
    kernel_size = offset.shape[1] // 2
    center = kernel_size // 2
    device = torch.device(device)

    y_offset_, x_offset_ = torch.split(offset, kernel_size, dim=1)

    y_center_ = torch.arange(0, width, dtype=torch.float32, device=device)
    y_center_ = einops.repeat(y_center_, "w -> k w h", k=kernel_size, h=height)

    x_center_ = torch.arange(0, height, dtype=torch.float32, device=device)
    x_center_ = einops.repeat(x_center_, "h -> k w h", k=kernel_size, w=width)

    if morph == 0:
        """
        Initialize the kernel and flatten the kernel
            y: only need 0
            x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
        """
        y_spread_ = torch.zeros([kernel_size], device=device)
        x_spread_ = torch.linspace(-center, center, kernel_size, device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        y_offset_ = einops.rearrange(y_offset_, "b k w h -> k b w h")
        y_offset_new_ = y_offset_.detach().clone()

        # The center position remains unchanged and the rest of the positions begin to swing
        # This part is quite simple. The main idea is that "offset is an iterative process"

        y_offset_new_[center] = 0

        for index in range(1, center + 1):
            y_offset_new_[center + index] = (
                y_offset_new_[center + index - 1] + y_offset_[center + index]
            )
            y_offset_new_[center - index] = (
                y_offset_new_[center - index + 1] + y_offset_[center - index]
            )

        y_offset_new_ = einops.rearrange(y_offset_new_, "k b w h -> b k w h")

        y_new_ = y_new_.add(y_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b (w k) h")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b (w k) h")

    elif morph == 1:
        """
        Initialize the kernel and flatten the kernel
            y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
            x: only need 0
        """
        y_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        x_spread_ = torch.zeros([kernel_size], device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        x_offset_ = einops.rearrange(x_offset_, "b k w h -> k b w h")
        x_offset_new_ = x_offset_.detach().clone()

        # The center position remains unchanged and the rest of the positions begin to swing
        # This part is quite simple. The main idea is that "offset is an iterative process"

        x_offset_new_[center] = 0

        for index in range(1, center + 1):
            x_offset_new_[center + index] = (
                x_offset_new_[center + index - 1] + x_offset_[center + index]
            )
            x_offset_new_[center - index] = (
                x_offset_new_[center - index + 1] + x_offset_[center - index]
            )

        x_offset_new_ = einops.rearrange(x_offset_new_, "k b w h -> b k w h")

        x_new_ = x_new_.add(x_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b w (h k)")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b w (h k)")

    return y_coordinate_map, x_coordinate_map


def get_interpolated_feature(
    input_feature: torch.Tensor,
    y_coordinate_map: torch.Tensor,
    x_coordinate_map: torch.Tensor,
    interpolate_mode: str = "bilinear",
):
    """From coordinate map interpolate feature of DSCNet based on: TODO

    Args:
        input_feature: feature that to be interpolated with shape [B, C, H, W]
        y_coordinate_map: coordinate map along y-axis with shape [B, K_H * H, K_W * W]
        x_coordinate_map: coordinate map along x-axis with shape [B, K_H * H, K_W * W]
        interpolate_mode: the arg 'mode' of nn.functional.grid_sample, can be 'bilinear' or 'bicubic' . Defaults to 'bilinear'.

    Return:
        interpolated_feature: interpolated feature with shape [B, C, K_H * H, K_W * W]
    """

    if interpolate_mode not in ("bilinear", "bicubic"):
        raise ValueError("interpolate_mode should be 'bilinear' or 'bicubic'.")

    y_max = input_feature.shape[-2] - 1
    x_max = input_feature.shape[-1] - 1

    y_coordinate_map_ = _coordinate_map_scaling(y_coordinate_map, origin=[0, y_max])
    x_coordinate_map_ = _coordinate_map_scaling(x_coordinate_map, origin=[0, x_max])

    y_coordinate_map_ = torch.unsqueeze(y_coordinate_map_, dim=-1)
    x_coordinate_map_ = torch.unsqueeze(x_coordinate_map_, dim=-1)

    # Note here grid with shape [B, H, W, 2]
    # Where [:, :, :, 2] refers to [x ,y]
    grid = torch.cat([x_coordinate_map_, y_coordinate_map_], dim=-1)

    interpolated_feature = nn.functional.grid_sample(
        input=input_feature,
        grid=grid,
        mode=interpolate_mode,
        padding_mode="zeros",
        align_corners=True,
    )

    return interpolated_feature


def _coordinate_map_scaling(
    coordinate_map: torch.Tensor,
    origin: list,
    target: list = [-1, 1],
):
    """Map the value of coordinate_map from origin=[min, max] to target=[a,b] for DSCNet based on: TODO

    Args:
        coordinate_map: the coordinate map to be scaled
        origin: original value range of coordinate map, e.g. [coordinate_map.min(), coordinate_map.max()]
        target: target value range of coordinate map,Defaults to [-1, 1]

    Return:
        coordinate_map_scaled: the coordinate map after scaling
    """
    min, max = origin
    a, b = target

    coordinate_map_scaled = torch.clamp(coordinate_map, min, max)

    scale_factor = (b - a) / (max - min)
    coordinate_map_scaled = a + scale_factor * (coordinate_map_scaled - min)

    return coordinate_map_scaled

class EncoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class DecoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)

        return x


class DSCNet(BaseModel):
    def __init__(
        self,
        n_channels,
        n_classes,
        kernel_size=9,
        extend_scope=1.0,
        if_offset=True,
        device='cuda',
        number=16,
        dim=1,
    ):
        """
        Our DSCNet
        :param n_channels: input channel
        :param n_classes: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        :param device: set on gpu
        :param number: basic layer numbers
        :param dim:
        """
        super().__init__()
        device = device
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.relu = nn.ReLU(inplace=True)
        self.number = number
        """
        The three contributions proposed in our paper are relatively independent. 
        In order to facilitate everyone to use them separately, 
        we first open source the network part of DSCNet. 
        <dim> is a parameter used by multiple templates, 
        which we will open source in the future ...
        """
        self.dim = dim  # This version dim is set to 1 by default, referring to a group of x-axes and y-axes
        """
        Here is our framework. Since the target also has non-tubular structure regions, 
        our designed model also incorporates the standard convolution kernel, 
        for fairness, we also add this operation to compare with other methods (like: Deformable Convolution).
        """
        self.conv00 = EncoderConv(n_channels, self.number)
        self.conv0x = DSConv_pro(
            n_channels,
            self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv0y = DSConv_pro(
            n_channels,
            self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv1 = EncoderConv(3 * self.number, self.number)

        self.conv20 = EncoderConv(self.number, 2 * self.number)
        self.conv2x = DSConv_pro(
            self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv2y = DSConv_pro(
            self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv3 = EncoderConv(6 * self.number, 2 * self.number)

        self.conv40 = EncoderConv(2 * self.number, 4 * self.number)
        self.conv4x = DSConv_pro(
            2 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv4y = DSConv_pro(
            2 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv5 = EncoderConv(12 * self.number, 4 * self.number)

        self.conv60 = EncoderConv(4 * self.number, 8 * self.number)
        self.conv6x = DSConv_pro(
            4 * self.number,
            8 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv6y = DSConv_pro(
            4 * self.number,
            8 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv7 = EncoderConv(24 * self.number, 8 * self.number)

        self.conv120 = EncoderConv(12 * self.number, 4 * self.number)
        self.conv12x = DSConv_pro(
            12 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv12y = DSConv_pro(
            12 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv13 = EncoderConv(12 * self.number, 4 * self.number)

        self.conv140 = DecoderConv(6 * self.number, 2 * self.number)
        self.conv14x = DSConv_pro(
            6 * self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv14y = DSConv_pro(
            6 * self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv15 = DecoderConv(6 * self.number, 2 * self.number)

        self.conv160 = DecoderConv(3 * self.number, self.number)
        self.conv16x = DSConv_pro(
            3 * self.number,
            self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv16y = DSConv_pro(
            3 * self.number,
            self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv17 = DecoderConv(3 * self.number, self.number)

        self.out_conv = nn.Conv2d(self.number, n_classes, 1)
        self.maxpooling = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # block0
        x_00_0 = self.conv00(x)
        x_0x_0 = self.conv0x(x)
        x_0y_0 = self.conv0y(x)
        x_0_1 = self.conv1(torch.cat([x_00_0, x_0x_0, x_0y_0], dim=1))

        # block1
        x = self.maxpooling(x_0_1)
        x_20_0 = self.conv20(x)
        x_2x_0 = self.conv2x(x)
        x_2y_0 = self.conv2y(x)
        x_1_1 = self.conv3(torch.cat([x_20_0, x_2x_0, x_2y_0], dim=1))

        # block2
        x = self.maxpooling(x_1_1)
        x_40_0 = self.conv40(x)
        x_4x_0 = self.conv4x(x)
        x_4y_0 = self.conv4y(x)
        x_2_1 = self.conv5(torch.cat([x_40_0, x_4x_0, x_4y_0], dim=1))

        # block3
        x = self.maxpooling(x_2_1)
        x_60_0 = self.conv60(x)
        x_6x_0 = self.conv6x(x)
        x_6y_0 = self.conv6y(x)
        x_3_1 = self.conv7(torch.cat([x_60_0, x_6x_0, x_6y_0], dim=1))

        # block4
        x = self.up(x_3_1)
        x_120_2 = self.conv120(torch.cat([x, x_2_1], dim=1))
        x_12x_2 = self.conv12x(torch.cat([x, x_2_1], dim=1))
        x_12y_2 = self.conv12y(torch.cat([x, x_2_1], dim=1))
        x_2_3 = self.conv13(torch.cat([x_120_2, x_12x_2, x_12y_2], dim=1))

        # block5
        x = self.up(x_2_3)
        x_140_2 = self.conv140(torch.cat([x, x_1_1], dim=1))
        x_14x_2 = self.conv14x(torch.cat([x, x_1_1], dim=1))
        x_14y_2 = self.conv14y(torch.cat([x, x_1_1], dim=1))
        x_1_3 = self.conv15(torch.cat([x_140_2, x_14x_2, x_14y_2], dim=1))

        # block6
        x = self.up(x_1_3)
        x_160_2 = self.conv160(torch.cat([x, x_0_1], dim=1))
        x_16x_2 = self.conv16x(torch.cat([x, x_0_1], dim=1))
        x_16y_2 = self.conv16y(torch.cat([x, x_0_1], dim=1))
        x_0_3 = self.conv17(torch.cat([x_160_2, x_16x_2, x_16y_2], dim=1))
        # x = self.dropout(x)
        out = self.out_conv(x_0_3)
        out = self.sigmoid(out)

        return out
