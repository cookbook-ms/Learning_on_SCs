#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build the SNN, gnn, psnn, scnn models by stacking the snn_conv layers with nonlinearity

"""

import sys
sys.path.append(".")
sys.path.append("..")

import numpy as np
import torch
import torchmetrics
import torch.nn as nn

from snn_conv import snn_conv
from psnn_conv import psnn_conv 
from scnn_conv import scnn_conv
from gnn_conv import gnn_conv

class scnn(nn.Module):
    def __init__(self, F_in, F_intermediate, F_out, K, K1, K2, laplacian, laplacian_l, laplacian_u, sigma, model_name):
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
        super(scnn, self).__init__()
        self.num_features = [F_in] + [F_intermediate[l] for l in range(len(F_intermediate))] + [F_out] # number of features vector e.g., [1 5 5 5 1]
        self.num_layers = len(self.num_features) 
        self.K = K
        self.K1 = K1
        self.K2 = K2 
        self.L = laplacian
        self.L_l = laplacian_l
        self.L_u = laplacian_u 
        self.sigma = sigma 
        nn_layer = []
        # define the NN layer operations for each model
        if model_name == 'snn':
            for l in range(self.num_layers-1):
                hyperparameters = {"F_in":self.num_features[l],"F_out":self.num_features[l+1],"K":self.K,"laplacian":self.L}
                nn_layer.extend([snn_conv(**hyperparameters), self.sigma])
                
        elif model_name == 'gnn':
            for l in range(self.num_layers-1):
                hyperparameters = {"F_in":self.num_features[l],"F_out":self.num_features[l+1],"K":self.K,"laplacian":self.L}
                nn_layer.extend([gnn_conv(**hyperparameters), self.sigma])
                
        elif model_name == 'scnn': 
            for l in range(self.num_layers-1):
                hyperparameters = {"F_in":self.num_features[l],"F_out":self.num_features[l+1],"K1":self.K1, "K2":self.K2,"laplacian_l":self.L_l,"laplacian_u":self.L_u}
                nn_layer.extend([scnn_conv(**hyperparameters), self.sigma])

        elif model_name == 'psnn':
            for l in range(self.num_layers-1): 
                hyperparameters = {
                    "F_in":self.num_features[l], "F_out":self.num_features[l+1],"laplacian_l":self.L_l,"laplacian_u":self.L_u
                }
                nn_layer.extend([psnn_conv(**hyperparameters), self.sigma])
        else: 
            raise Exception('invalid model type')
        
        self.simplicial_nn = nn.Sequential(*nn_layer)

    def forward(self,x):
        return self.simplicial_nn(x)#.view(-1,1).T