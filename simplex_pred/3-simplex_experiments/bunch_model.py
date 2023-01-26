#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build the bunch architecture convolution

paper: https://arxiv.org/abs/2012.06010
"""

import sys
sys.path.append(".")
sys.path.append("..")

import numpy as np
import torch.nn as nn

from bunch_conv import bunch_conv

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
    
class bunch(nn.Module):
    def __init__(self, F_in, F_intermediate, F_out, b2, b3, l1, l2, l3, d1,d2,d3,d5, sigma, model_name):
        super(bunch, self).__init__()
        self.num_features = [F_in] + [F_intermediate[l] for l in range(len(F_intermediate))] + [F_out] # number of features vector e.g., [1 5 5 5 1]
        self.num_layers = len(self.num_features) 

        self.b2 = b2 
        self.b3 = b3
        self.l1 = l1 
        self.l2 = l2
        self.l3 = l3
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d5 = d5
        self.sigma = sigma 
        nn_layer = []
        # define the NN layer operations for each model
        if model_name in ['bunch_tri', 'bunch_edge']:
            for l in range(self.num_layers-1):
                hyperparameters = {"F_in":self.num_features[l],"F_out":self.num_features[l+1],"b2":self.b2, "b3":self.b3, "l1":self.l1, "l2":self.l2, "l3":self.l3,"d1":self.d1, "d2":self.d2, "d3":self.d3, "d5":self.d5,"sigma":self.sigma}
                nn_layer.extend([bunch_conv(**hyperparameters)])        
    
        else: 
            raise Exception('invalid model type')
        
        self.simplicial_nn = nn.Sequential(*nn_layer)

    def forward(self,x_in):
        return self.simplicial_nn(x_in)#.view(-1,1).T