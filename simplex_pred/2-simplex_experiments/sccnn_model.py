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

from sccnn_conv import sccnn_conv,sccnn_conv_sc_1
'''for ablations below'''
from sccnn_conv import sccnn_conv_no_n_to_e,sccnn_conv_no_e_to_e,sccnn_conv_no_t_to_e,sccnn_conv_no_n_to_n

from sccnn_conv import sccnn_conv_no_n_to_e_sc_1,sccnn_conv_no_e_to_e_sc_1,sccnn_conv_no_e_to_n_sc_1,sccnn_conv_no_n_to_n_sc_1


class sccnn(nn.Module):
    def __init__(self, F_in, F_intermediate, F_out, b1, b2, l0, l1l, l1u, l2, d1,d2,d3,d5, k00,k0p,k1n,k11,k12,k1p,k2n,k22, sigma, model_name):
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
            if model_name in ['sccnn_node','sccnn_edge','sccnn_node_missing_node','sccnn_node_missing_edge','sccnn_node_missing_node_edge']: # original sccnn for sc of order two, and some ablation study: limited input data 
                nn_layer.extend([sccnn_conv(**hyperparameters)]) 
            elif model_name in ['sccnn_node_sc_1','sccnn_edge_sc_1']: # no b2: sc of order one
                nn_layer.extend([sccnn_conv_sc_1(**hyperparameters)])
            ### ablation for sc order 2 
            elif model_name in ['sccnn_node_no_t_to_e']: # no triangle to edge contribution
                nn_layer.extend([sccnn_conv_no_t_to_e(**hyperparameters)])  
            elif model_name in ['sccnn_node_no_n_to_e']: # no node to edge
                nn_layer.extend([sccnn_conv_no_n_to_e(**hyperparameters)])  
            elif model_name in ['sccnn_node_no_e_to_e']: # no edge to edge
                nn_layer.extend([sccnn_conv_no_e_to_e(**hyperparameters)])  
            elif model_name in ['sccnn_node_no_n_to_n']: # no node to node
                nn_layer.extend([sccnn_conv_no_n_to_n(**hyperparameters)])   
            ##### ablation for sc order 1    
            elif model_name in ['sccnn_node_no_n_to_n_sc_1']: # no b2, no node to node 
                nn_layer.extend([sccnn_conv_no_n_to_n_sc_1(**hyperparameters)]) 
            elif model_name in ['sccnn_node_no_e_to_e_sc_1']: # no b2, no edge to edge
                nn_layer.extend([sccnn_conv_no_e_to_e_sc_1(**hyperparameters)])
            elif model_name in ['sccnn_node_no_e_to_n_sc_1']: # no b2, no edge to node
                nn_layer.extend([sccnn_conv_no_e_to_n_sc_1(**hyperparameters)]) 
            elif model_name in ['sccnn_node_no_n_to_e_sc_1']: # no b2, no node to edge
                nn_layer.extend([sccnn_conv_no_n_to_e_sc_1(**hyperparameters)])  
            else: 
                raise Exception('invalid model type')
        
        self.simplicial_nn = nn.Sequential(*nn_layer)

    def forward(self,x):
        return self.simplicial_nn(x)#.view(-1,1).T