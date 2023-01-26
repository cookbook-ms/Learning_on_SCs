#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
linear filters corresponding to sccnn, i.e., cf-sc

https://arxiv.org/abs/2201.12584

'''
import sys
sys.path.append(".")
sys.path.append("..")

import numpy as np
import torch
import torchmetrics
import torch.nn as nn

from sccnn_conv import sccnn_conv_id

class sccnn(nn.Module):
    def __init__(self, F_in, F_intermediate, F_out, b1, b2, b3, l0, l1l, l1u, l2l, l2u, l3, d1,d2,d3,d5, d10, d20, d30, k00, k0p,k1n,k11,k12,k1p, k2n,k21,k22,k2p,k3n,k33, sigma, model_name):
        """
        Parameters
        ----------
        - F_in: number of the input features : 1
        - F_intermediate: number of intermediate features per 
        """
        super(sccnn, self).__init__()
        self.num_features = [F_in] + [F_intermediate[l] for l in range(len(F_intermediate))] + [F_out] # number of features vector e.g., [1 5 5 5 1]
        self.num_layers = len(self.num_features) 
        
        self.b1 = b1
        self.b2 = b2 
        self.b3 = b3
        
        self.l0 = l0 
        self.l1l = l1l
        self.l1u = l1u
        self.l2l = l2l 
        self.l2u = l2u
        self.l3 = l3
        
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d5 = d5
        
        self.d10 = d10
        self.d20 = d20
        self.d30 = d30
        
        self.k00 = k00
        self.k0p = k0p
        
        self.k1n = k1n 
        self.k11 = k11
        self.k12 = k12
        self.k1p = k1p
        
        self.k2n = k2n
        self.k21 = k21
        self.k22 = k22
        self.k2p = k2p
        
        self.k3n = k3n
        self.k33 = k33

        
        self.sigma = sigma 
        nn_layer = []
        if model_name in ['sccnn_node', 'sccnn_edge', 'sccnn_tri']:
            for l in range(self.num_layers-1):
                hyperparameters = {"F_in":self.num_features[l],"F_out":self.num_features[l+1],"b1":self.b1, "b2":self.b2, "b3":self.b3, "l0":self.l0, "l1l":self.l1l, "l1u":self.l1u, "l2l":self.l2l, "l2u":self.l2u, "l3":self.l3, "d1":self.d1, "d2":self.d2, "d3":self.d3, "d5":self.d5,"d10":self.d10, "d20":self.d20, "d30":self.d30, "k00":k00, "k0p":k0p, "k1n":k1n,  "k11":k11, "k12":k12, "k1p":k1p, "k2n":k2n, "k21":k21, "k22":k22, "k2p":k2p, "k3n":k3n, "k33":k33, "sigma":self.sigma}
                nn_layer.extend([sccnn_conv_id(**hyperparameters)]) 
        else: 
            raise Exception('invalid model type')
        
        self.simplicial_nn = nn.Sequential(*nn_layer)

    def forward(self,x):
        return self.simplicial_nn(x)#.view(-1,1).T