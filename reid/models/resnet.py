from __future__ import absolute_import

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable
import torchvision
# from torch_deform_conv.layers import ConvOffset2D
from reid.utils.serialization import load_checkpoint, save_checkpoint

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, checkpoint=None, pretrained=True, num_features=2048, 
                    dropout=0.1, num_classes=0):
        super(ResNet, self).__init__()

        self.depth = depth
        self.checkpoint = checkpoint
        self.pretrained = pretrained
        self.num_features = num_features
        self.dropout = dropout
        self.num_classes = num_classes

        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)
        out_planes = self.base.fc.in_features

        # resume from pre-iteration training 
        if self.checkpoint:
            state_dict = load_checkpoint(checkpoint)
            self.load_state_dict(state_dict['state_dict'], strict=False)

        self.feat = nn.Linear(out_planes, self.num_features, bias=False)
        self.feat_bn = nn.BatchNorm1d(self.num_features)
        self.relu = nn.ReLU(inplace=True)
        init.normal(self.feat.weight, std=0.001)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)

        #x2 classifier
        self.classifier_x2 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.classifier_x2.weight, std=0.001)
        init.constant(self.classifier_x2.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        x1 = F.avg_pool2d(x, x.size()[2:])
        x1 = x1.view(x1.size(0), -1)
        x2 = F.avg_pool2d(x, x.size()[2:])
        x2 = x2.view(x2.size(0), -1)
        x2 = self.feat(x2)
        x2 = self.feat_bn(x2)
        x2 = self.relu(x2)
        return x1,x2

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)

