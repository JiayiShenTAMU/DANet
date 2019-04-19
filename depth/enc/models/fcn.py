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
        self.head_depth = FCNHead(2048, 1, norm_layer)
        self.parse_fcrn = nn.Sequential(
            nn.Conv2d(64, 19, 3, padding=1),
            nn.Upsample((768, 768), mode='bilinear'))
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x, rev = False):
        imsize = x.size()[2:]
        #d_out = self.fcrn(depth)
        #d_out = self.parse_fcrn(d_out)
        #print(d_out.shape)
        _, _, c3, c4 = self.base_forward(x)
        #print(c3.shape)
        x = self.head(c4, rev=rev)
        #x = upsample(x, imsize, **self._up_kwargs)
        x = upsample(x, imsize, **self._up_kwargs)
        #print("UPSAMPLE")
        x_depth = self.head_depth(c4, rev=rev)
        x_depth = upsample(x, imsize, **self._up_kwargs)
        #print(x.shape)
        #x[1] = upsample(x[1], imsize, **self._up_kwargs).view(-1, imsize[0], imsize[1])
        outputs = [x, x_depth]
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
                                   nn.Dropout2d(0.1, False))
        self.conv6 = nn.Conv2d(inter_channels, out_channels, 1)
        self.conv7 = nn.Conv2d(inter_channels, 1, 1)

    def forward(self, x, rev = False):
        #normal_out = self.conv5(x)
        #depth_out = self.conv6(x)
        #outputs = [normal_out]
        #outputs.append(depth_out)
        #return self.conv5(x)
        #return tuple(outputs)
        normal_out = self.conv5(x)
        if not rev:
            normal_out = self.conv6(normal_out)
        else:
            normal_out = self.conv7(normal_out)
        return normal_out


def get_fcn(dataset='pascal_voc', backbone='resnet50', pretrained=False,
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

def get_fcn_resnet50_pcontext(pretrained=False, root='~/.encoding/models', **kwargs):
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
    return get_fcn('pcontext', 'resnet50', pretrained, root=root, aux=False, **kwargs)

def get_fcn_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
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
    return get_fcn('ade20k', 'resnet50', pretrained, root=root, **kwargs)
