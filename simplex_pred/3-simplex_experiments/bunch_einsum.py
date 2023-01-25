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

from bunch_conv_einsum import bunch_conv

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
        """
        Parameters
        ----------
        - F_in: number of the input features : 1
        - F_intermediate: number of intermediate features per layer e.g., [2,5,5] -- 2 outputs in the 2nd layer, 5 outputs in the 3rd and 4th layer, but not including the last layer, which has again 1 output in general 
        - F_out: number of the output features: generally 1

        - K: filter order when using only one shift operator; when K1, K2 are applied, set K as none or 0
        - K1: filter order of the lower shift operator
        - K2: filter order of the upper shift operator

        - laplacian: the hodge laplacian of the corresponding order 
        - laplacian_l: the lower laplacian of the corresponding order 
        - laplacian_u: the upper laplacian of the corresponding order 

        - sigma: the chosen nonlinearity, e.g., nn.LeakyReLU()
        - alpha_leaky_relu: the negative slope of the leakyrelu, if applied

        - model: choose the architecture - "snn", "scnn", "psnn" 
        """
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