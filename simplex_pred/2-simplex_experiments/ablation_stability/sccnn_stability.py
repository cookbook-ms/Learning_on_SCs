#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
equivalent to sccnn model but for the purpose of study stability 
"""

import sys
sys.path.append(".")
sys.path.append("..")

import numpy as np
import torch
import torchmetrics
import torch.nn as nn

from sccnn_conv import sccnn_conv_stability


class sccnn_stability(nn.Module):
    def __init__(self, F_in, F_intermediate, F_out, b1, b2, l0, l1l, l1u, l2, d1,d2,d3,d4,d5, k00,k0p,k1n,k11,k12,k1p,k2n,k22, sigma, model_name):
        super(sccnn_stability, self).__init__()
        self.num_features = [F_in] + [F_intermediate[l] for l in range(len(F_intermediate))] + [F_out] # number of features vector e.g., [1 5 5 5 1]
        self.num_layers = len(self.num_features) 
        self.b1 = b1
        self.b2 = b2 
        self.l0 = l0 
        self.l1l = l1l
        self.l1u = l1u
        self.l2 = l2
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4 
        self.d5 = d5

        self.sigma = sigma 
        nn_layer = []
        print(model_name)
        assert model_name == 'sccnn_node_stability'

        for l in range(self.num_layers-1):
            hyperparameters = {"F_in":self.num_features[l],"F_out":self.num_features[l+1],"b1":self.b1, "b2":self.b2, "l0":self.l0, "l1l":self.l1l, "l1u":self.l1u, "l2":self.l2, "d1":self.d1, "d2":self.d2, "d3":self.d3, "d4":self.d4, "d5":self.d5, "k00":k00, "k0p":k0p, "k1n":k1n,  "k11":k11, "k12":k12, "k1p":k1p, "k2n":k2n, "k22":k22, "sigma":self.sigma}
            nn_layer.extend([sccnn_conv_stability(**hyperparameters)])
        
        self.simplicial_nn = nn.Sequential(*nn_layer)

    def forward(self,x):
        return self.simplicial_nn(x)#.view(-1,1).T