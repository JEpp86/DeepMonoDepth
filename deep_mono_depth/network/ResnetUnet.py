"""!
@file Resnet-UNet.py
@brief Resnet Based model for Baseline.
@author Jason Epp
"""
import torch
import torch.nn as nn
import torchvision.models

from enum import Enum, auto


def convelu(in_channels, out_channels, kernel, padding):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
    nn.ELU(inplace=True)
  )

def convrelu(in_channels, out_channels, kernel, padding):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
    nn.ReLU(inplace=True),
  )

def convsig(in_channels, out_channels, kernel):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel, padding=(int(kernel//2))),
    nn.Sigmoid()
  )

class ResnetModel(Enum):
  RESNET_18 = auto()
  RESNET_34 = auto()


class ResNetUNet(nn.Module):


  def __init__(self, model_type: ResnetModel):
    super().__init__()

    # Dowload base model
    if model_type == ResnetModel.RESNET_18:
        self.base_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    elif model_type == ResnetModel.RESNET_34:
        self.base_model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
    else:
        raise("Unsupported Model Type")
    self.base_layers = list(self.base_model.children())

    # split model into a series of layers
    self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
    self.layer0_1x1 = convelu(64, 64, 1, 0)
    self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
    self.layer1_1x1 = convelu(64, 64, 1, 0)
    self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
    self.layer2_1x1 = convelu(128, 128, 1, 0)
    self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
    self.layer3_1x1 = convelu(256, 256, 1, 0)
    self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
    self.layer4_1x1 = convelu(512, 512, 1, 0)

    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    self.conv_up3 = convelu(256 + 512, 512, 3, 1)
    self.conv_up2 = convelu(128 + 512, 256, 3, 1)
    self.conv_up1 = convelu(64 + 256, 256, 3, 1)
    self.conv_up0 = convelu(64 + 256, 128, 3, 1)

    self.conv_original_size0 = convelu(3, 64, 3, 1)
    self.conv_original_size1 = convelu(64, 64, 3, 1)
    self.conv_original_size2 = convelu(64 + 128, 64, 3, 1)

    self.conv_s3 = convsig(256, 1, 3)
    self.conv_s2 = convsig(256, 1, 3)
    self.conv_s1 = convsig(128, 1, 3)
    self.conv_last = convsig(64, 1, 3) #nn.Conv2d(64, 1, 1)

  def forward(self, input):

    # monodepth normalization only on input
    input = (input - 0.45) /  0.225

    x_original = self.conv_original_size0(input)
    x_original = self.conv_original_size1(x_original)

    layer0 = self.layer0(input)
    layer1 = self.layer1(layer0)
    layer2 = self.layer2(layer1)
    layer3 = self.layer3(layer2)
    layer4 = self.layer4(layer3)

    layer4 = self.layer4_1x1(layer4)
    x = self.upsample(layer4)
    layer3 = self.layer3_1x1(layer3)
    x = torch.cat([x, layer3], dim=1)
    x = self.conv_up3(x)

    x = self.upsample(x)
    layer2 = self.layer2_1x1(layer2)
    x = torch.cat([x, layer2], dim=1)
    x3 = self.conv_up2(x)

    x = self.upsample(x3)
    layer1 = self.layer1_1x1(layer1)
    x = torch.cat([x, layer1], dim=1)
    x2 = self.conv_up1(x)

    x = self.upsample(x2)
    layer0 = self.layer0_1x1(layer0)
    x = torch.cat([x, layer0], dim=1)
    x1 = self.conv_up0(x)

    x = self.upsample(x1)
    x = torch.cat([x, x_original], dim=1)
    x = self.conv_original_size2(x)

    # Multiscale output
    out3 = self.conv_s3(x3)
    out2 = self.conv_s2(x2)
    out1 = self.conv_s1(x1)
    out0 = self.conv_last(x)

    return out0, out1, out2, out3

