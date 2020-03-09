import torch
from torch import nn
from torchvision import models


class vgg16_feat(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_layers = models.vgg16(pretrained=True).features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return output


class perceptual_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']

    def forward(self, *args, **kwargs):
        x1_feat = args[0]
        x2_feat = args[1]

        loss = 0
        for key in self.names:
            size = x1_feat[key].size()
            # L1 loss
            # loss += (x1_feat[key] - x2_feat[key]).abs().sum() / (size[0] * size[1] * size[2] * size[3])
            # MSE loss
            loss += ((x1_feat[key] - x2_feat[key]) ** 2).sum() / (size[0] * size[1] * size[2] * size[3])

        loss /= 4
        return loss