#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build the SCNN convolution

paper: https://arxiv.org/abs/2110.02585
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from chebyshev import chebyshev

class scnn_conv(nn.Module):
    def __init__(self, F_in, F_out, K1, K2, laplacian_l, laplacian_u):
        """
        F_in: number of input features per layer 
        F_out: number of output features per layer
        K1, K2: the filter lower and upper orders, on the lower and upper laplacians respectively
        alpha_leaky_relu: the negative slop of the leaky relu function 
        """
        super(scnn_conv, self).__init__()
        self.F_in = F_in
        self.F_out = F_out 
        self.K1 = K1
        self.K2 = K2 
        
        self.W = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 1+self.K1+self.K2))) 
        self.Ll = laplacian_l
        self.Lu = laplacian_u
        self.dout = nn.Dropout(p=0.0)
        
        dim_simp = self.Ll.size(dim=0)
        self.I = torch.eye(dim_simp,device=self.Ll.device)
        
        self.reset_parameters()
        print("created SCNN layers")

    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W.data, gain=gain)

    def forward(self,x):
        """
        define the simplicial convolution in the SCNN architecture (i.e., the subspace-varying simplicial filtering operation)
        x: input features of dimension M x F_in (num_edges/simplices x num_input features)
        """
        Ix = torch.unsqueeze(self.I@x,2)
        if self.K1 > 0 and self.K2 > 0:
            Xl = chebyshev(self.Ll, self.K1, x)
            Xu = chebyshev(self.Lu, self.K2, x)
            X = torch.cat((Ix,Xl,Xu),2)
        elif self.K1 > 0 and self.K2 == 0:
            Xl = chebyshev(self.Ll, self.K1, x)
            X = torch.cat((Ix,Xl),2)
        elif self.K2 > 0 and self.K1 == 0:
            Xu = chebyshev(self.Lu, self.K2, x)
            X = torch.cat((Ix,Xu),2)
        else:
            X = Ix

        y = torch.einsum('nik,iok->no',X,self.W)
        y = self.dout(y)
        return y