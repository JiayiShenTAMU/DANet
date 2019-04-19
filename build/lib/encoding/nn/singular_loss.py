from __future__ import absolute_import
from __future__ import division

import logging

import torch
import torch.nn as nn

import os

from .cross_entropy_loss import CrossEntropyLoss


class SingularLoss(nn.Module):

    def __init__(self, *, use_gpu=True, beta=None, penalty_position='before'):
        super().__init__()

        os_beta = None

        sing_beta = os.environ.get('sing_beta')
        try:
            os_beta = float(sing_beta)
        except (ValueError, TypeError):
            os_beta = 1e-6

        self.USE_LOG = False
        print('USE_GPU', use_gpu)
        self.beta = os_beta
        print('O.F. Beta', self.beta)
        self.penalty_position = frozenset(penalty_position.split(','))

    def dominant_eigenvalue(self, A):

        B, N, _ = A.size()
        x = torch.randn(B, N, 1, device='cuda')

        for _ in range(1):
            x = torch.bmm(A, x)
        x: 'B x N x 1'
        numerator = torch.bmm(
            torch.bmm(A, x).view(B, 1, N),
            x
        ).squeeze()
        denominator = (torch.norm(x.view(B, N), p=2, dim=1) ** 2).squeeze()
        # denominator = torch.norm(
        #     x.view(B, 1, N),
        #     x
        # ).squeeze()
        # # print(denominator)

        return numerator / denominator

    def get_singular_values(self, A):

        AAT = torch.bmm(A, A.permute(0, 2, 1))
        B, N, _ = AAT.size()
        largest = self.dominant_eigenvalue(AAT)
        I = torch.eye(N, device='cuda').expand(B, N, N)  # noqa
        I = I * largest.view(B, 1, 1).repeat(1, N, N)  # noqa
        tmp = self.dominant_eigenvalue(AAT - I)
        return tmp + largest, largest

    def apply_penalty(self, k, x):

        if isinstance(x, tuple):
            return sum([self.apply_penalty(k, xx) for xx in x]) / len(x)

        batches, channels, height, width = x.size()
        W = x.view(batches, channels, -1)
        smallest, largest = self.get_singular_values(W)
        if not self.USE_LOG:
            singular_penalty = (largest - smallest) * self.beta
        else:
            singular_penalty = (torch.log1p(largest) - torch.log1p(smallest)) * self.beta

        if k == 'layer5':
            singular_penalty *= 0.01

        return singular_penalty.sum() / (x.size()[0] / 32.)  # Quirk: normalize to 32-batch case

    def forward(self, inputs, target):

        _, feature_dict = tuple(inputs)

        existed_positions = frozenset(feature_dict.keys())
        missing = self.penalty_position - existed_positions
        if missing:
            raise RuntimeError('Cannot apply singular loss, as positions {!r} are missing.'.format(list(missing)))

        singular_penalty = sum([self.apply_penalty(k, x) for k, x in feature_dict.items() if k in self.penalty_position])

        return singular_penalty