#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build the bunch model by stacking the bunch_conv layers with nonlinearity

paper: https://arxiv.org/abs/2012.06010
"""

import sys
sys.path.append(".")
sys.path.append("..")

import numpy as np
import torch.nn as nn

from bunch_conv import bunch_conv,bunch_conv_sc_1
    
class bunch(nn.Module):
    def __init__(self, F_in, F_intermediate, F_out, b1, b2, l0, l1, l2, d1,d2,d3,d5, sigma, model_name):
        super(bunch, self).__init__()
        self.num_features = [F_in] + [F_intermediate[l] for l in range(len(F_intermediate))] + [F_out] # number of features vector e.g., [1 5 5 5 1]
        self.num_layers = len(self.num_features) 
        self.b1 = b1
        self.b2 = b2 
        self.l0 = l0 
        self.l1 = l1
        self.l2 = l2
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d5 = d5
        
        self.sigma = sigma 
        nn_layer = []
        # define the NN layer operations for each model
        if model_name in ['bunch_node', 'bunch_edge']:
            for l in range(self.num_layers-1):
                hyperparameters = {"F_in":self.num_features[l],"F_out":self.num_features[l+1],"b1":self.b1, "b2":self.b2, "l0":self.l0, "l1":self.l1, "l2":self.l2, "d1":self.d1, "d2":self.d2, "d3":self.d3, "d5":self.d5,"sigma":self.sigma}
                nn_layer.extend([bunch_conv(**hyperparameters)])        
        elif model_name in ['bunch_node_sc_1', 'bunch_edge_sc_1']:
            for l in range(self.num_layers-1):
                hyperparameters = {"F_in":self.num_features[l],"F_out":self.num_features[l+1],"b1":self.b1, "b2":self.b2, "l0":self.l0, "l1":self.l1, "l2":self.l2, "d1":self.d1, "d2":self.d2, "d3":self.d3, "d5":self.d5,"sigma":self.sigma}
                nn_layer.extend([bunch_conv_sc_1(**hyperparameters)])
        else: 
            raise Exception('invalid model type')
        
        self.simplicial_nn = nn.Sequential(*nn_layer)

    def forward(self,x_in):
        return self.simplicial_nn(x_in)#.view(-1,1).T