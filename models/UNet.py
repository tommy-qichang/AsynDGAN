import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from parse_config import ConfigParser

class dilated_conv(nn.Module):
    """ same as original conv if dilation equals to 1 """
    def __init__(self, in_channel, out_channel, kernel_size=3, dropout_rate=0.0, activation=F.relu, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm2d(out_channel)
        self.activation = activation
        if dropout_rate > 0:
            self.drop = nn.Dropout2d(p=dropout_rate)
        else:
            self.drop = lambda x: x  # no-op

    def forward(self, x):
        # CAB: conv -> activation -> batch normal
        x = self.norm(self.activation(self.conv(x)))
        x = self.drop(x)
        return x


class ConvDownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.0, dilation=1):
        super().__init__()
        self.conv1 = dilated_conv(in_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)
        self.conv2 = dilated_conv(out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.pool(x), x


class ConvUpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.0, dilation=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, 2, stride=2)
        self.conv1 = dilated_conv(in_channel // 2 + out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)
        self.conv2 = dilated_conv(out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)

    def forward(self, x, x_skip):
        x = self.up(x)
        H_diff = x.shape[2] - x_skip.shape[2]
        W_diff = x.shape[3] - x_skip.shape[3]
        x_skip = F.pad(x_skip, (0, W_diff, 0, H_diff), mode='reflect')
        x = torch.cat([x, x_skip], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_c=3, out_c=2):
        super().__init__()
        # down conv
        self.c1 = ConvDownBlock(in_c, 64)
        self.c2 = ConvDownBlock(64, 128)
        self.c3 = ConvDownBlock(128, 256)
        self.c4 = ConvDownBlock(256, 512)
        self.cu = ConvDownBlock(512, 1024)
        # up conv
        self.u5 = ConvUpBlock(1024, 512)
        self.u6 = ConvUpBlock(512, 256)
        self.u7 = ConvUpBlock(256, 128)
        self.u8 = ConvUpBlock(128, 64)
        # final conv
        self.ce = nn.Conv2d(64, out_c, kernel_size=1)

    def forward(self, x):
        x, c1 = self.c1(x)
        x, c2 = self.c2(x)
        x, c3 = self.c3(x)
        x, c4 = self.c4(x)
        _, x = self.cu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.u5(x, c4)
        x = self.u6(x, c3)
        x = self.u7(x, c2)
        x = self.u8(x, c1)
        x = self.ce(x)
        return x


# Transfer Learning ResNet as Encoder part of UNet
class ResUNet34(nn.Module):
    def __init__(self, out_c=2, pretrained=True, fixed_feature=False):
        super().__init__()
        # load weight of pre-trained resnet
        self.resnet = models.resnet34(pretrained=pretrained)
        if fixed_feature:
            for param in self.resnet.parameters():
                param.requires_grad = False
        # up conv
        l = [64, 64, 128, 256, 512]
        self.u5 = ConvUpBlock(l[4], l[3], dropout_rate=0.1)
        self.u6 = ConvUpBlock(l[3], l[2], dropout_rate=0.1)
        self.u7 = ConvUpBlock(l[2], l[1], dropout_rate=0.1)
        self.u8 = ConvUpBlock(l[1], l[0], dropout_rate=0.1)
        # final conv
        self.ce = nn.ConvTranspose2d(l[0], out_c, 2, stride=2)
        # self.ce_sigma = nn.ConvTranspose2d(l[0], out_c, 2, stride=2)

    def forward(self, x):
        # refer https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = c1 = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = c2 = self.resnet.layer1(x)
        x = c3 = self.resnet.layer2(x)
        x = c4 = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.u5(x, c4)
        x = self.u6(x, c3)
        x = self.u7(x, c2)
        x = self.u8(x, c1)
        x1 = self.ce(x)
        # x2 = self.ce_sigma(x)
        return x1

def setup_unet():
    unet = UNet(in_c=3, out_c=1)
    resume_path = "/share_hd1/projects/code_seg_v2/experiments/seg/real_imgs_unet/models/All_seg_train/1022_121541/model_best.pth"
    checkpoint = torch.load(resume_path)
    unet.load_state_dict(checkpoint['state_dict'])
    unet.eval()
    return unet
