"""
ERFNet-based encoder-decodersegmentation model for lane detection
Copied and modified from voldemortX/pytorch-auto-drive

Architecture:
Image (B, 3, H(288), W(800))
    |
    v
Encoder - downsample 8x, extract features -> (B, 128, H/8(36), W/8(100))
    |
    v
Decoder - upsample 8x -> (B, num_classes(5), H(288), W(800))
EDLaneExist - conv(128->32) -> conv(32->5) -> softmax -> pool -> flatten -> linear(3965->128) -> linear(128->5) -> (B, 4) lane existence confidence

A standard 3x3 conv is replaced by two sequential factorized convolutions (1x3 followed by 3x1) to reduce parameters and increase efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from sas.models.segmentation.non_bottleneck_1d import non_bottleneck_1d
from ..builder import MODELS

class Downsampler(nn.Module):
    """halves the spatial dimensions and doubles the number of channels"""
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class Encoder(nn.Module):
    def __init__(self, num_classes, dropout_1=0.03, dropout_2=0.3):
        super().__init__()
        # 1. Initial block
        self.initial_block = Downsampler(3, 16) # 3 input channels, 16 output channels, H/2

        self.layers = nn.ModuleList()

        # 2. Downsampler + 5x non_bottleneck_1d(dilation=1) for local features
        self.layers.append(Downsampler(16, 64)) # 16 input channels, 64 output channels, H/4

        for x in range(0, 5):
            self.layers.append(non_bottleneck_1d(64, dropout_1, 1)) # 5 residual blocks, no dialtion
        
        # 3. Downsampler + 8x non_bottleneck_1d(dilation=2,4,8,16) for larger context via dilation
        self.layers.append(Downsampler(64, 128)) # 64 input channels, 128 output channels, H/8

        #  8 dilated blocks repated 2 times
        for x in range(0, 2):
            self.layers.append(non_bottleneck_1d(128, dropout_2, 2))
            self.layers.append(non_bottleneck_1d(128, dropout_2, 4))
            self.layers.append(non_bottleneck_1d(128, dropout_2, 8))
            self.layers.append(non_bottleneck_1d(128, dropout_2, 16))

        # Only in encoder mode
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            # raw (128, H/8, W/8) feature map output from encoder
            output = layer(output)

        if predict:
            # normally false
            output = self.output_conv(output)

        return output


class UpsamplerBlock(nn.Module):
    """upsamples by a factor of 2"""
    def __init__(self, ninput, noutput):
        super().__init__()
        # Double spatial resolution with stride 2
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):
    """upsamples the feature map back to input resolution"""
    def __init__(self, num_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(UpsamplerBlock(128, 64)) # H/8 -> H/4
        self.layers.append(non_bottleneck_1d(64, 0, 1)) # refine, no dropout, no dilation
        self.layers.append(non_bottleneck_1d(64, 0, 1)) # refine, no dropout, no dilation
        self.layers.append(UpsamplerBlock(64, 16)) # H/4 -> H/2
        self.layers.append(non_bottleneck_1d(16, 0, 1)) # refine, no dropout, no dilation
        self.layers.append(non_bottleneck_1d(16, 0, 1)) # refine, no dropout, no dilation
        
        # Output (num_classes, H, W) - one channel per class
        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True) # H/2 -> H

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


class _EncoderDecoderModel(nn.Module):
    """Base class for encoder-decoder segmentation models"""
    def __init__(self):
        super().__init__()

    def _load_encoder(self, pretrained_weights):
        if pretrained_weights is not None:
            try:
                saved_weights = torch.load(pretrained_weights)['state_dict']
            except FileNotFoundError:
                print(f"Pretrained weights not found at {pretrained_weights}, loading failed.")
                return
            original_weights = self.state_dict()
            for key in saved_weights.keys():
                if key in original_weights.keys():
                    original_weights[key] = saved_weights[key]
            self.load_state_dict(original_weights)
        else:
            print("No pretrained weights provided, loading failed.")

    def forward(self, x):
        pass


@MODELS.register()
class ERFNet(_EncoderDecoderModel):
    def __init__(self,
                num_classes,
                lane_exist_cfg=None,
                spatial_conv_cfg=None,
                dropout_1=0.03,
                dropout_2=0.3,
                pretrained_weights=None
                ):
        super().__init__()
        self.encoder = Encoder(num_classes, dropout_1, dropout_2)
        self.decoder = Decoder(num_classes)
        # Optional SCNN (spatial message passing)
        self.spatial_conv = MODELS.from_dict(spatial_conv_cfg)
        # EDExist head
        self.lane_exist = MODELS.from_dict(lane_exist_cfg)
        self._load_encoder(pretrained_weights)

    def _load_encoder(self, pretrained_weights):
        if pretrained_weights is not None:
            try:
                saved_weights = torch.load(pretrained_weights)['state_dict']
            except FileNotFoundError:
                print(f"Pretrained weights not found at {pretrained_weights}, loading failed.")
                return

            original_weights = self.state_dict()
            for key in saved_weights.keys():
                my_key = key.replace('module.features.', '') # remove "module.features." prefix from ERFNet weights
                if my_key in original_weights.keys():
                    original_weights[my_key] = saved_weights[key]
            self.load_state_dict(original_weights)
        else:
            print("No pretrained weights provided, loading failed.")

    def forward(self, x, only_encode=False):
        out = OrderedDict()
        if only_encode:
            return self.encoder.forward(x, predict=True)
        else:
            output = self.encoder(x) # (B, 128, H/8, W/8)
            if self.spatial_conv is not None:
                output = self.spatial_conv(output)  # SCNN message passing
            out['out'] = self.decoder.forward(output)   # (B, num_classes, H, W) segmentation output

            if self.lane_exist is not None:
                out['lane'] = self.lane_exist(output)  # (B, num_classes-1) lane existence classification

            return out


@MODELS.register()
class EDLaneExist(nn.Module):
    """
    Lane existence head for ERFNet/Enet

    Lane ordering is indexd from left to right from cam view
    CULane
    cam view:   L2  L1  Car R1  R2
    seg class:  1   2       3   4   (class 0 is background)
    lane idx:   0   1       2   3
    ego lane boundary is 1 and 2

    TuSimple(num_classes=7)
    6 lane lines
    ego lane boundary is 3 and 4

    Input:
    - (128, H/8, W/8) feature map from encoder
    Output:
    - (num_classes-1, H/8, W/8) confidence map for lane existence
    """
    def __init__(self, num_output, flattened_size=3965, dropout=0.1, pool='avg'):
        super().__init__()

        self.layers = nn.ModuleList()
        # 128 -> 32 channels -> BN -> ReLU
        self.layers.append(nn.Conv2d(128, 32, (3, 3), stride=1, padding=(4, 4), bias=False, dilation=(4, 4)))
        self.layers.append(nn.BatchNorm2d(32, eps=1e-03))

        self.layers_final = nn.ModuleList()
        self.layers_final.append(nn.Dropout2d(dropout))
        # 32 -> 5 channels -> Softmax(dim=1), 5 channel confidence map
        self.layers_final.append(nn.Conv2d(32, 5, (1, 1), stride=1, padding=0, bias=True))

        # halve spatial size
        if pool == 'max':
            self.pool = nn.MaxPool2d(2, stride=2)
        elif pool == 'avg':
            self.pool = nn.AvgPool2d(2, stride=2)
        else:
            raise RuntimeError(f"Unsupported pool type {pool}")

        self.linear1 = nn.Linear(flattened_size, 128)
        self.linear2 = nn.Linear(128, num_output)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)

        output = F.relu(output)

        for layer in self.layers_final:
            output = layer(output)

        output = F.softmax(output, dim=1)
        output = self.pool(output)
        output = output.flatten(start_dim=1)
        output = self.linear1(output)
        output = F.relu(output)
        output = self.linear2(output)

        return output