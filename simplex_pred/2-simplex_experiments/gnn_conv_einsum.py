#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build the GNN convolution

paper: https://arxiv.org/abs/1606.09375
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from chebyshev import chebyshev


class gnn_conv(nn.Module):
    
    def __init__(self, F_in, F_out, K, laplacian):
        """
        F_in: number of input features per layer 
        F_out: number of output features per layer
        """
        super(gnn_conv, self).__init__()
        self.F_in = F_in
        self.F_out = F_out 
        self.K = K 

        # define the filter weights, which is of dimension K x F_in x F_out
        self.W = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.K)))  
        
        self.L = laplacian 
        
        dim_simp = self.L.size(dim=0)
        self.I = torch.eye(dim_simp,device=self.L.get_device())
        
        self.reset_parameters()
        print("created GNN layers")

    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W.data, gain=gain)

    def forward(self,x):
        """
        define the simplicial convolution in the SNN architecture (i.e., the simplicial filtering operation)
        x: input features of dimension M x F_in (num_edges/simplices x num_input features)
        """
        Ix = torch.unsqueeze(self.I@x,2)
        if self.K > 0:
            X = chebyshev(self.L, self.K, x)
            X = torch.cat((Ix,X),2)
        else:
            X = Ix 
        y = torch.einsum('nik,iok->no',X,self.W)

        return y