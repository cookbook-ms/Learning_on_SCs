#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build the PSNN architecture convolution

paper: https://arxiv.org/pdf/2102.10058.pdf
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F

class psnn_conv(nn.Module):
    def __init__(self, F_in, F_out, laplacian_l, laplacian_u):
        """
        F_in: number of input features per layer 
        F_out: number of output features per layer
        """
        super(psnn_conv, self).__init__()
        self.F_in = F_in
        self.F_out = F_out 
        
        self.W0 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+1+1)))
        self.Ll = laplacian_l
        self.Lu = laplacian_u
        dim_simp = self.Ll.size(dim=0)
        self.I = torch.eye(dim_simp,device=self.Ll.get_device())
        self.reset_parameters()
        print("created PSNN layers")
    
    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W0.data, gain=gain)
        
    def forward(self,x):
        Ix = torch.unsqueeze(self.I@x,2)
        Llx = torch.unsqueeze(self.Ll@x,2)
        Lux = torch.unsqueeze(self.Lu@x,2)
        X = torch.cat((Ix, Llx, Lux),2)
        y = torch.einsum('nik,iok->no',X,self.W0)

        return y 