###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample
from .fcrn import FCRN
from .weights import load_weights
dtype = torch.cuda.FloatTensor
from .base import BaseNet
import torchvision.models as models

__all__ = ['FCN', 'get_fcn', 'get_fcn_resnet50_pcontext', 'get_fcn_resnet50_ade']

class FCN(BaseNet):
    r"""Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    Examples
    --------
    >>> model = FCN(nclass=21, backbone='resnet50')
    >>> print(model)
    """
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(FCN, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = FCNHead(2048, nclass, norm_layer)
        self.fcrn = FCRN(1)
        self.fcrn.load_state_dict(load_weights(self.fcrn, "NYU_ResNet-UpProj.npy", dtype))
        self.fcrn.load_state_dict(torch.load('checkpoint.pth.tar')['state_dict'])
        self.fcrn.train()
        #torch.save(self.pretrained, "temp.pkl")
        #self.depth = torch.load('temp.pkl')
        #self.depth.load_state_dict(self.pretrained.state_dict())
        self.depth.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def depth_forward(self, x):
        x = self.depth.conv1(x)
        x = self.depth.bn1(x)
        x = self.depth.relu(x)
        x = self.depth.maxpool(x)
        c1 = self.depth.layer1(x)
        c2 = self.depth.layer2(c1)
        c3 = self.depth.layer3(c2)
        c4 = self.depth.layer4(c3)
        return c1, c2, c3, c4

    def rgb_forward(self, x, d1, d2, d3, d4):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        
        c1 = torch.add(c1, d1)
        c2 = self.pretrained.layer2(c1)
        
        c2 = torch.add(c2, d2)
        c3 = self.pretrained.layer3(c2)
        
        c3 = torch.add(c3, d3)
        c4 = self.pretrained.layer4(c3)
        
        c4 = torch.add(c4, d4)
        return c1, c2, c3, c4

    def forward(self, x, depth):
        imsize = x.size()[2:]
        d_out = self.fcrn(depth)
        #print(d_out.shape)
        
        d1, d2, d3, d4 = self.depth_forward(d_out)
        c1, c2, c3, c4 = self.rgb_forward(x, d1, d2, d3, d4)
        #print(c3.shape)
        x = self.head(c4)
        #x = upsample(x, imsize, **self._up_kwargs)
        x = upsample(x, imsize, **self._up_kwargs)
        #print("UPSAMPLE")
        #x += d_out
        #print(x.shape)
        #x[1] = upsample(x[1], imsize, **self._up_kwargs).view(-1, imsize[0], imsize[1])
        outputs = [x, x]
        #outputs.append(x[1])
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = upsample(auxout, imsize, **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)

        
class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

        self.conv6 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, 1, 1))
    def forward(self, x):
        #normal_out = self.conv5(x)
        #depth_out = self.conv6(x)
        #outputs = [normal_out]
        #outputs.append(depth_out)
        #return self.conv5(x)
        #return tuple(outputs)
        normal_out = self.conv5(x)
        return normal_out


def get_fcn_with_fuse(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='/raid/sunfangwen/wzh/.torch/models', **kwargs):
    r"""FCN model from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf>`_
    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    Examples
    --------
    >>> model = get_fcn(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'pcontext': 'pcontext',
        'ade20k': 'ade',
    }
    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = FCN(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)),
            strict= False)
    return model

def get_fcn_with_fuse_resnet50_pcontext(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_fcn_resnet50_pcontext(pretrained=True)
    >>> print(model)
    """
    return get_fcn_with_fuse('pcontext', 'resnet50', pretrained, root=root, aux=False, **kwargs)

def get_fcn_with_fuse_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_fcn_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_fcn_with_fuse('ade20k', 'resnet50', pretrained, root=root, **kwargs)
