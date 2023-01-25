#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sccnn model together with ablation study 
"""

import sys
sys.path.append(".")
sys.path.append("..")

import numpy as np
import torch
import torchmetrics
import torch.nn as nn

from sccnn_conv_einsum import sccnn_conv,sccnn_conv_no_x2,sccnn_conv_no_x0,sccnn_conv_no_x1,sccnn_conv_ebli,sccnn_conv_no_x0_to_node,sccnn_conv_no_b2,sccnn_conv_no_x0_sc1,sccnn_conv_no_x0_to_node_sc1,sccnn_conv_no_x1_sc1,sccnn_conv_no_x1_to_node_sc1,sccnn_conv_stability


class sccnn(nn.Module):
    def __init__(self, F_in, F_intermediate, F_out, b1, b2, l0, l1l, l1u, l2, d1,d2,d3,d5, k00,k0p,k1n,k11,k12,k1p,k2n,k22, sigma, model_name):
        """
        Parameters
        ----------
        - F_in: number of the input features : 1
        - F_intermediate: number of intermediate features per layer e.g., [2,5,5] -- 2 outputs in the 2nd layer, 5 outputs in the 3rd and 4th layer, but not including the last layer, which has again 1 output in general 
        - F_out: number of the output features: generally 1

        - K: filter orders for all scfs

        - sigma: the chosen nonlinearity, e.g., nn.LeakyReLU()
        - alpha_leaky_relu: the negative slope of the leakyrelu, if applied

        - model: choose the architecture - "snn", "scnn", "psnn" 
        """
        super(sccnn, self).__init__()
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
        self.d5 = d5

        self.sigma = sigma 
        nn_layer = []
        print(model_name)
        
        for l in range(self.num_layers-1):
            hyperparameters = {"F_in":self.num_features[l],"F_out":self.num_features[l+1],"b1":self.b1, "b2":self.b2, "l0":self.l0, "l1l":self.l1l, "l1u":self.l1u, "l2":self.l2, "d1":self.d1, "d2":self.d2, "d3":self.d3, "d5":self.d5, "k00":k00, "k0p":k0p, "k1n":k1n,  "k11":k11, "k12":k12, "k1p":k1p, "k2n":k2n, "k22":k22, "sigma":self.sigma}
            if model_name in ['sccnn_node','sccnn_edge','sccnn_node_no_lower','sccnn_node_no_upper','sccnn_node_no_conv_node','sccnn_node_zero_node','sccnn_node_zero_edge','sccnn_node_zero_all_input','sccnn_node_missing_node','sccnn_node_missing_edge','sccnn_node_missing_node_edge']: # original sccnn for sc of order two, and some ablation study: including sccnn_node without lower or upper convolutions in the edge space, no convolution in the node space, and limited input data 
                nn_layer.extend([sccnn_conv(**hyperparameters)]) 
            elif model_name in ['sccnn_node_no_tri','sccnn_edge_no_tri']: # no triangle to edge contribution
                nn_layer.extend([sccnn_conv_no_x2(**hyperparameters)])  
            elif model_name in ['sccnn_node_no_node','sccnn_edge_no_node']: # no node to edge
                nn_layer.extend([sccnn_conv_no_x0(**hyperparameters)])  
            elif model_name in ['sccnn_node_no_edge','sccnn_edge_no_edge']: # no edge to edge
                nn_layer.extend([sccnn_conv_no_x1(**hyperparameters)])  
            elif model_name in ['sccnn_node_no_node_to_node']: # no node to node
                nn_layer.extend([sccnn_conv_no_x0_to_node(**hyperparameters)])   
            elif model_name in ['sccnn_node_no_b2','sccnn_edge_no_b2']: # no b2: sc of order one
                nn_layer.extend([sccnn_conv_no_b2(**hyperparameters)]) 
            elif model_name in ['sccnn_node_no_b2_no_node_to_node']: # no b2, no node to node 
                nn_layer.extend([sccnn_conv_no_x0_to_node_sc1(**hyperparameters)]) 
            elif model_name in ['sccnn_node_no_b2_no_edge_to_edge']: # no b2, no edge to edge
                nn_layer.extend([sccnn_conv_no_x1_sc1(**hyperparameters)])
            elif model_name in ['sccnn_node_no_b2_no_edge_to_node']: # no b2, no edge to node
                nn_layer.extend([sccnn_conv_no_x1_to_node_sc1(**hyperparameters)]) 
            elif model_name in ['sccnn_edge_no_b2_no_node_to_edge','sccnn_node_no_b2_no_node_to_edge']: # no b2, no node to edge
                nn_layer.extend([sccnn_conv_no_x0_sc1(**hyperparameters)])  
            elif model_name == 'sccnn_node_ebli':
                nn_layer.extend([sccnn_conv_ebli(**hyperparameters)])  
            else: 
                raise Exception('invalid model type')
        
        self.simplicial_nn = nn.Sequential(*nn_layer)

    def forward(self,x):
        return self.simplicial_nn(x)#.view(-1,1).T