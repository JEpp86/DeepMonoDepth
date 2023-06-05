"""!
@file Resnet-PoseNet.py
@brief Resnet Based model for determinig pose between inputs.
@author Jason Epp
"""
import torch
import torch.nn as nn
import torchvision.models

class ResNetPoseNet(nn.Module):
  def __init__(self):
    super().__init__()

    # Dowload base model
    self.base_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT) #pretrained=True) 
    self.base_layers = list(self.base_model.children())

    # concatenate input weihgts to make 6 channel input (two concatenated images)
    self.base_layers[0].weight = torch.nn.Parameter(torch.cat((self.base_layers[0].weight, self.base_layers[0].weight), dim=1))
    self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
    self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
    self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
    self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
    self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
    self.squeeze = nn.Conv2d(512, 256, 1)
    self.pose0 = nn.Conv2d(256, 256, 3, 1, 1)
    self.pose1 = nn.Conv2d(256, 256, 3, 1, 1)
    self.pose2 = nn.Conv2d(256,12,1)

    self.relu = nn.ReLU()

  def forward(self, input):

    # monodepth normalization only on input
    input = (input - 0.45) /  0.225

    layer0 = self.layer0(input)
    layer1 = self.layer1(layer0)
    layer2 = self.layer2(layer1)
    layer3 = self.layer3(layer2)
    layer4 = self.layer4(layer3)
    squeeze = self.relu(self.squeeze(layer4))
    pose = self.relu(self.pose0(squeeze))
    pose = self.relu(self.pose1(pose))
    pose = self.pose2(pose)

    pose = pose.mean(3).mean(2)
    # scale for stability
    pose = 0.01 * pose.view(-1, 2, 1, 6)
    axis = pose[...,:3]
    translation = pose[...,3:]
    return axis, translation

if __name__ == '__main__':
    net = ResNetPoseNet()
    in_data = torch.rand([4,6,380,1024])
    print(in_data.shape)
    out = net(in_data)
    print(out.shape)
