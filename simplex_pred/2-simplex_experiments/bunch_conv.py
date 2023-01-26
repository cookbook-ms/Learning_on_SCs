#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build the bunch architecture convolution

paper: https://arxiv.org/abs/2012.06010
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F


class bunch_conv(nn.Module):
    def __init__(self, F_in, F_out, b1, b2, l0, l1, l2, d1,d2,d3,d5, sigma):
        """
        F_in: number of input features per layer 
        F_out: number of output features per layer
        """
        super(bunch_conv, self).__init__()
        self.F_in = F_in
        self.F_out = F_out 
        self.B1 = b1 
        self.B2 = b2
        self.L0 = l0
        self.L1 = l1        
        self.L2 = l2
        self.D1 = d1
        self.D2 = d2
        self.D3 = d3
        self.D5 = d5
        self.sigma = sigma
        self.W0 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,2)))
        self.W1 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,3)))
        self.W2 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,2)))
        dim_simp = self.L1.size(dim=0)

        self.I = torch.eye(dim_simp,device=self.L1.device)
        self.reset_parameters()
        print("created Bunch layers")
    
    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W0.data, gain=gain)
        nn.init.xavier_uniform_(self.W1.data, gain=gain)
        nn.init.xavier_uniform_(self.W2.data, gain=gain)
        
    
    def forward(self,x_in):
        x0,x1,x2 = x_in
        x00 = torch.unsqueeze(self.L0@x0,2)
        x01 = torch.unsqueeze(torch.inverse(self.D1)@self.B1@x1,2)
        
        x10 = torch.unsqueeze(self.D2@self.B1.T @torch.inverse(self.D1)@ x0,2)
        x11 = torch.unsqueeze(self.L1 @ x1,2)
        x12 = torch.unsqueeze(self.B2 @self.D3@ x2,2)        
#        print(x10.size(),x11.size(),x12.size())

        x20 = torch.unsqueeze(self.B2.T@torch.pinverse(self.D5)@x1,2)
        x21 = torch.unsqueeze(self.L2 @ x2,2)    

        X0 = torch.cat((x00,x01),2)
        X1 = torch.cat((x10,x11,x12),2)
        X2 = torch.cat((x20,x21),2)
        y0 = torch.einsum('nik,iok->no',X0,self.W0)
        y1 = torch.einsum('nik,iok->no',X1,self.W1)
        y2 = torch.einsum('nik,iok->no',X2,self.W2)
        y0 = self.sigma(y0)
        y1 = self.sigma(y1)
        y2 = self.sigma(y2)
        return y0,y1,y2
    
class bunch_conv_sc_1(nn.Module):
    def __init__(self, F_in, F_out, b1, b2, l0, l1, l2, d1,d2,d3,d5, sigma):
        """
        F_in: number of input features per layer 
        F_out: number of output features per layer
        """
        super(bunch_conv_sc_1, self).__init__()
        self.F_in = F_in
        self.F_out = F_out 
        self.B1 = b1 
        self.B2 = b2
        self.L0 = l0
        self.L1 = l1        
        self.L2 = l2
        self.D1 = d1
        self.D2 = d2
        self.D3 = d3
        self.D5 = d5
        self.sigma = sigma
        self.W0 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,2)))
        self.W1 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,2)))
        # self.W2 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,2)))
        dim_simp = self.L1.size(dim=0)

        self.I = torch.eye(dim_simp,device=self.L1.device)
        self.reset_parameters()
        print("created Bunch layers")
    
    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W0.data, gain=gain)
        nn.init.xavier_uniform_(self.W1.data, gain=gain)
        # nn.init.xavier_uniform_(self.W2.data, gain=gain)
        
    
    def forward(self,x_in):
        x0,x1 = x_in
        x00 = torch.unsqueeze(self.L0@x0,2)
        x01 = torch.unsqueeze(torch.inverse(self.D1)@self.B1@x1,2)
        
        x10 = torch.unsqueeze(self.D2@self.B1.T @torch.inverse(self.D1)@ x0,2)
        x11 = torch.unsqueeze(self.L1 @ x1,2)     
#        print(x10.size(),x11.size(),x12.size())

        # x20 = torch.unsqueeze(self.B2.T@torch.pinverse(self.D5)@x1,2)
        # x21 = torch.unsqueeze(self.L2 @ x2,2)    

        X0 = torch.cat((x00,x01),2)
        X1 = torch.cat((x10,x11),2)

        y0 = torch.einsum('nik,iok->no',X0,self.W0)
        y1 = torch.einsum('nik,iok->no',X1,self.W1)

        y0 = self.sigma(y0)
        y1 = self.sigma(y1)

        return y0,y1
    