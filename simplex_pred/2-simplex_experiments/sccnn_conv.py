#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch 
import torch.nn as nn
import torch.nn.functional as F
from chebyshev import chebyshev

'''sccnn node, sccnn edge'''
class sccnn_conv(nn.Module):
    def __init__(self, F_in, F_out, b1, b2, l0, l1l, l1u, l2, d1,d2,d3,d5, k00,k0p,k1n,k11,k12,k1p,k2n,k22, sigma):
        """
        F_in: number of input features per layer 
        F_out: number of output features per layer
        p: stands for positive, denoting the upper simplex oreder
        n: stands for negative, denoting the lower simplex order
        """
        super(sccnn_conv, self).__init__()
        self.F_in = F_in
        self.F_out = F_out 
        self.B1 = b1 
        self.B2 = b2
        self.L0 = l0
        self.L1l = l1l
        self.L1u = l1u        
        self.L2 = l2
        self.D1 = d1
        self.D2 = d2
        self.D3 = d3
        self.D5 = d5
        self.sigma = sigma
        self.k00 = k00
        self.k0p = k0p
        self.k1n = k1n 
        self.k11 = k11
        self.k12 = k12
        self.k1p = k1p
        self.k2n = k2n
        self.k22 = k22
        
        self.W0 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k00 + 1+self.k0p)))
        self.W1 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k1n + 1+self.k11+self.k12 + 1+self.k1p)))
        self.W2 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 1+self.k2n + 1+self.k22)))
        
        dim_0 = self.L0.size(dim=0)
        self.I0 = torch.eye(dim_0,device=self.L0.device)
        dim_1 = self.L1l.size(dim=0)
        self.I1 = torch.eye(dim_1,device=self.L1l.device)
        dim_2 = self.L2.size(dim=0)
        self.I2 = torch.eye(dim_2,device=self.L2.device)
        
        
        self.reset_parameters()
        print("created SCCNN layers")
    
    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W0.data, gain=gain)
        nn.init.xavier_uniform_(self.W1.data, gain=gain)
        nn.init.xavier_uniform_(self.W2.data, gain=gain)
        
    
    def forward(self,x_in):
        x0,x1,x2 = x_in
        
        '''order 0 '''
        I0x = torch.unsqueeze(self.I0@x0,2)
        x0p = torch.inverse(self.D1)@self.B1@x1
        I0xp = torch.unsqueeze(self.I0@x0p,2) # torch.inverse(self.D1)@self.B1 is the projection matrix
        
        if self.k00 > 0 and self.k0p > 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,X00,I0xp,X0p),2)
        elif  self.k00 > 0 and self.k0p == 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0 = torch.cat((I0x, X00, I0xp ), 2)
        elif self.k00 > 0 and self.k0p > 0:
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,I0xp,X0p),2)
        else: 
            X0 = torch.cat((I0x,I0xp),2)

        '''order 1'''
        x1n = self.D2@self.B1.T @torch.inverse(self.D1)@ x0
        I1xn = torch.unsqueeze(self.I1@x1n,2)
        I1x = torch.unsqueeze(self.I1@x1,2)
        x1p = self.B2 @self.D3@ x2
        I1xp = torch.unsqueeze(self.I1@x1p,2)
        
        if self.k1n > 0:
            X1nl = chebyshev(self.L1l, self.k1n, x1n)
            X1n = torch.cat((I1xn,X1nl),2)
        else: 
            X1n = I1xn
        
        if self.k11>0 and self.k12>0:
            X1l = chebyshev(self.L1l, self.k11, x1)
            X1u = chebyshev(self.L1u, self.k12, x1)
            X11 = torch.cat((I1x, X1l, X1u),2)
        elif self.k11>0 and self.k12==0:
            X1l = chebyshev(self.L1l, self.k11, x1)
            X11 = torch.cat((I1x, X1l),2)  
        elif self.k11==0 and self.k12>0:
            X1u = chebyshev(self.L1u, self.k12, x1)
            X11 = torch.cat((I1x, X1u),2)
        else:
            X11 = I1x
            
        if self.k1p > 0:
            X1pu = chebyshev(self.L1u, self.k1p, x1p)
            X1p = torch.cat((I1xp, X1pu), 2)
        else:
            X1p = I1xp
            
        X1 = torch.cat((X1n,X11,X1p),2)
            
        '''order 2'''
        x2n = self.B2.T@torch.pinverse(self.D5)@x1
        I2xn = torch.unsqueeze(self.I2@x2n,2)
        I2x = torch.unsqueeze(self.I2@x2,2)
        if self.k2n>0 and self.k22 > 0:
            X2n = chebyshev(self.L2, self.k2n, x2n)
            X22 = chebyshev(self.L2, self.k22, x2)
            X2 = torch.cat((I2xn, X2n, I2x, X22), 2)
        elif self.k2n>0 and self.k22 == 0:
            X2n = chebyshev(self.L2, self.k2n, x2n)
            X2 = torch.cat((I2xn, X2n, I2x), 2)
        elif self.k2n==0 and self.k22 > 0:
            X22 = chebyshev(self.L2, self.k22, x2)
            X2 = torch.cat((I2xn, I2x, X22), 2)   
        else:
            X2 = torch.cat((I2xn,I2x),2)

        y0 = torch.einsum('nik,iok->no',X0,self.W0)
        y1 = torch.einsum('nik,iok->no',X1,self.W1)
        y2 = torch.einsum('nik,iok->no',X2,self.W2)
        y0 = self.sigma(y0)
        y1 = self.sigma(y1)
        y2 = self.sigma(y2)
        return y0,y1,y2

'''no B2 is defined, i.e., sc of order one '''    
class sccnn_conv_sc_1(nn.Module):
    def __init__(self, F_in, F_out, b1, b2, l0, l1l, l1u, l2, d1,d2,d3,d5, k00,k0p,k1n,k11,k12,k1p,k2n,k22, sigma):
        """
        F_in: number of input features per layer 
        F_out: number of output features per layer
        p: stands for positive, denoting the upper simplex oreder
        n: stands for negative, denoting the lower simplex order
        """
        super(sccnn_conv_sc_1, self).__init__()
        self.F_in = F_in
        self.F_out = F_out 
        self.B1 = b1 
        self.B2 = b2
        self.L0 = l0
        self.L1l = l1l
        self.L1u = l1u        
        self.L2 = l2
        self.D1 = d1
        self.D2 = d2
        self.D3 = d3
        self.D5 = d5
        self.sigma = sigma
        self.k00 = k00
        self.k0p = k0p
        self.k1n = k1n 
        self.k11 = k11
        self.k12 = k12
        self.k1p = k1p
        self.k2n = k2n
        self.k22 = k22
        
        self.W0 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k00 + 1+self.k0p)))
        self.W1 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k1n + 1+self.k11)))
        #self.W2 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 1+self.k2n + 1+self.k22)))
        
        dim_0 = self.L0.size(dim=0)
        self.I0 = torch.eye(dim_0,device=self.L0.device)
        dim_1 = self.L1l.size(dim=0)
        self.I1 = torch.eye(dim_1,device=self.L1l.device)
        # dim_2 = self.L2.size(dim=0)
        # self.I2 = torch.eye(dim_2,device=self.L2.device)
        
        self.reset_parameters()
        print("created SCCNN layers but no triangles included")
    
    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W0.data, gain=gain)
        nn.init.xavier_uniform_(self.W1.data, gain=gain)
        #nn.init.xavier_uniform_(self.W2.data, gain=gain)
        
    
    def forward(self,x_in):
        x0,x1 = x_in
        
        '''order 0 '''
        I0x = torch.unsqueeze(self.I0@x0,2)
        x0p = torch.inverse(self.D1)@self.B1@x1
        I0xp = torch.unsqueeze(self.I0@x0p,2) # torch.inverse(self.D1)@self.B1 is the projection matrix
        
        if self.k00 > 0 and self.k0p > 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,X00,I0xp,X0p),2)
        elif  self.k00 > 0 and self.k0p == 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0 = torch.cat((I0x, X00, I0xp ), 2)
        elif self.k00 > 0 and self.k0p > 0:
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,I0xp,X0p),2)
        else: 
            X0 = torch.cat((I0x,I0xp),2)

        '''order 1'''
        x1n = self.D2@self.B1.T @torch.inverse(self.D1)@ x0
        I1xn = torch.unsqueeze(self.I1@x1n,2)
        I1x = torch.unsqueeze(self.I1@x1,2)
        # x1p = self.B2 @self.D3@ x2
        # I1xp = torch.unsqueeze(self.I1@x1p,2)
        
        if self.k1n > 0:
            X1nl = chebyshev(self.L1l, self.k1n, x1n)
            X1n = torch.cat((I1xn,X1nl),2)
        else: 
            X1n = I1xn
        
        if self.k11>0 and self.k12>0:
            X1l = chebyshev(self.L1l, self.k11, x1)
        #    X1u = chebyshev(self.L1u, self.k12, x1)
            X11 = torch.cat((I1x, X1l),2)
        elif self.k11>0 and self.k12==0:
            X1l = chebyshev(self.L1l, self.k11, x1)
            X11 = torch.cat((I1x, X1l),2)  
        # elif self.k11==0 and self.k12>0:
        #     X1u = chebyshev(self.L1u, self.k12, x1)
        #     X11 = torch.cat((I1x, X1u),2)
        else:
            X11 = I1x
            
        # if self.k1p > 0:
        #     X1pu = chebyshev(self.L1u, self.k1p, x1p)
        #     X1p = torch.cat((I1xp, X1pu), 2)
        # else:
        #     X1p = I1xp
            
        X1 = torch.cat((X1n,X11),2)
            
        # '''order 2'''
        # x2n = self.B2.T@torch.pinverse(self.D5)@x1
        # I2xn = torch.unsqueeze(self.I2@x2n,2)
        # I2x = torch.unsqueeze(self.I2@x2,2)
        # if self.k2n>0 and self.k22 > 0:
        #     X2n = chebyshev(self.L2, self.k2n, x2n)
        #     X22 = chebyshev(self.L2, self.k22, x2)
        #     X2 = torch.cat((I2xn, X2n, I2x, X22), 2)
        # elif self.k2n>0 and self.k22 == 0:
        #     X2n = chebyshev(self.L2, self.k2n, x2n)
        #     X2 = torch.cat((I2xn, X2n, I2x), 2)
        # elif self.k2n==0 and self.k22 > 0:
        #     X22 = chebyshev(self.L2, self.k22, x2)
        #     X2 = torch.cat((I2xn, I2x, X22), 2)   
        # else:
        #     X2 = torch.cat((I2xn,I2x),2)

        y0 = torch.einsum('nik,iok->no',X0,self.W0)
        y1 = torch.einsum('nik,iok->no',X1,self.W1)
        # y2 = torch.einsum('nik,iok->no',X2,self.W2)
        y0 = self.sigma(y0)
        y1 = self.sigma(y1)
        # y2 = self.sigma(y2)
        return y0,y1#,y2

'''filter bank'''
class sccnn_conv_id(nn.Module):
    def __init__(self, F_in, F_out, b1, b2, l0, l1l, l1u, l2, d1,d2,d3,d5, k00,k0p,k1n,k11,k12,k1p,k2n,k22, sigma):
        """
        F_in: number of input features per layer 
        F_out: number of output features per layer
        p: stands for positive, denoting the upper simplex oreder
        n: stands for negative, denoting the lower simplex order
        """
        super(sccnn_conv_id, self).__init__()
        self.F_in = F_in
        self.F_out = F_out 
        self.B1 = b1 
        self.B2 = b2
        self.L0 = l0
        self.L1l = l1l
        self.L1u = l1u        
        self.L2 = l2
        self.D1 = d1
        self.D2 = d2
        self.D3 = d3
        self.D5 = d5
        self.sigma = sigma
        self.k00 = k00
        self.k0p = k0p
        self.k1n = k1n 
        self.k11 = k11
        self.k12 = k12
        self.k1p = k1p
        self.k2n = k2n
        self.k22 = k22
        
        self.W0 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k00 + 1+self.k0p)))
        self.W1 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k1n + 1+self.k11+self.k12 + 1+self.k1p)))
        self.W2 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 1+self.k2n + 1+self.k22)))
        
        dim_0 = self.L0.size(dim=0)
        self.I0 = torch.eye(dim_0,device=self.L0.device)
        dim_1 = self.L1l.size(dim=0)
        self.I1 = torch.eye(dim_1,device=self.L1l.device)
        dim_2 = self.L2.size(dim=0)
        self.I2 = torch.eye(dim_2,device=self.L2.device)
        
        
        self.reset_parameters()
        print("created SCCNN layers")
    
    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W0.data, gain=gain)
        nn.init.xavier_uniform_(self.W1.data, gain=gain)
        nn.init.xavier_uniform_(self.W2.data, gain=gain)
        
    
    def forward(self,x_in):
        x0,x1,x2 = x_in
        
        '''order 0 '''
        I0x = torch.unsqueeze(self.I0@x0,2)
        x0p = torch.inverse(self.D1)@self.B1@x1
        I0xp = torch.unsqueeze(self.I0@x0p,2) # torch.inverse(self.D1)@self.B1 is the projection matrix
        
        if self.k00 > 0 and self.k0p > 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,X00,I0xp,X0p),2)
        elif  self.k00 > 0 and self.k0p == 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0 = torch.cat((I0x, X00, I0xp ), 2)
        elif self.k00 > 0 and self.k0p > 0:
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,I0xp,X0p),2)
        else: 
            X0 = torch.cat((I0x,I0xp),2)

        '''order 1'''
        x1n = self.D2@self.B1.T @torch.inverse(self.D1)@ x0
        I1xn = torch.unsqueeze(self.I1@x1n,2)
        I1x = torch.unsqueeze(self.I1@x1,2)
        x1p = self.B2 @self.D3@ x2
        I1xp = torch.unsqueeze(self.I1@x1p,2)
        
        if self.k1n > 0:
            X1nl = chebyshev(self.L1l, self.k1n, x1n)
            X1n = torch.cat((I1xn,X1nl),2)
        else: 
            X1n = I1xn
        
        if self.k11>0 and self.k12>0:
            X1l = chebyshev(self.L1l, self.k11, x1)
            X1u = chebyshev(self.L1u, self.k12, x1)
            X11 = torch.cat((I1x, X1l, X1u),2)
        elif self.k11>0 and self.k12==0:
            X1l = chebyshev(self.L1l, self.k11, x1)
            X11 = torch.cat((I1x, X1l),2)  
        elif self.k11==0 and self.k12>0:
            X1u = chebyshev(self.L1u, self.k12, x1)
            X11 = torch.cat((I1x, X1u),2)
        else:
            X11 = I1x
            
        if self.k1p > 0:
            X1pu = chebyshev(self.L1u, self.k1p, x1p)
            X1p = torch.cat((I1xp, X1pu), 2)
        else:
            X1p = I1xp
            
        X1 = torch.cat((X1n,X11,X1p),2)
            
        '''order 2'''
        x2n = self.B2.T@torch.pinverse(self.D5)@x1
        I2xn = torch.unsqueeze(self.I2@x2n,2)
        I2x = torch.unsqueeze(self.I2@x2,2)
        if self.k2n>0 and self.k22 > 0:
            X2n = chebyshev(self.L2, self.k2n, x2n)
            X22 = chebyshev(self.L2, self.k22, x2)
            X2 = torch.cat((I2xn, X2n, I2x, X22), 2)
        elif self.k2n>0 and self.k22 == 0:
            X2n = chebyshev(self.L2, self.k2n, x2n)
            X2 = torch.cat((I2xn, X2n, I2x), 2)
        elif self.k2n==0 and self.k22 > 0:
            X22 = chebyshev(self.L2, self.k22, x2)
            X2 = torch.cat((I2xn, I2x, X22), 2)   
        else:
            X2 = torch.cat((I2xn,I2x),2)

        y0 = torch.einsum('nik,iok->no',X0,self.W0)
        y1 = torch.einsum('nik,iok->no',X1,self.W1)
        y2 = torch.einsum('nik,iok->no',X2,self.W2)
 
        return y0,y1,y2
    

'''convolutions below can be used for ablation study'''

'''no B1.T@x0 in x1: no node to edge'''
class sccnn_conv_no_n_to_e(nn.Module):
    def __init__(self, F_in, F_out, b1, b2, l0, l1l, l1u, l2, d1,d2,d3,d5, k00,k0p,k1n,k11,k12,k1p,k2n,k22, sigma):
        """
        F_in: number of input features per layer 
        F_out: number of output features per layer
        p: stands for positive, denoting the upper simplex oreder
        n: stands for negative, denoting the lower simplex order
        """
        super(sccnn_conv_no_n_to_e, self).__init__()
        self.F_in = F_in
        self.F_out = F_out 
        self.B1 = b1 
        self.B2 = b2
        self.L0 = l0
        self.L1l = l1l
        self.L1u = l1u        
        self.L2 = l2
        self.D1 = d1
        self.D2 = d2
        self.D3 = d3
        self.D5 = d5
        self.sigma = sigma
        self.k00 = k00
        self.k0p = k0p
        self.k1n = k1n 
        self.k11 = k11
        self.k12 = k12
        self.k1p = k1p
        self.k2n = k2n
        self.k22 = k22
        
        self.W0 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 1+self.k00 + 1+self.k0p))) #only for projection from x1 to x0
        self.W1 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 1+self.k11+self.k12 + 1+self.k1p)))
        self.W2 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 1+self.k2n + 1+self.k22)))
        
        dim_0 = self.L0.size(dim=0)
        self.I0 = torch.eye(dim_0,device=self.L0.device)
        dim_1 = self.L1l.size(dim=0)
        self.I1 = torch.eye(dim_1,device=self.L1l.device)
        dim_2 = self.L2.size(dim=0)
        self.I2 = torch.eye(dim_2,device=self.L2.device)
        
        
        self.reset_parameters()
        print("created SCCNN layers but no nodes included")
    
    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W0.data, gain=gain)
        nn.init.xavier_uniform_(self.W1.data, gain=gain)
        nn.init.xavier_uniform_(self.W2.data, gain=gain)
        
    
    def forward(self,x_in):
        x0,x1,x2 = x_in
        # '''order 0 '''
        I0x = torch.unsqueeze(self.I0@x0,2)
        x0p = torch.inverse(self.D1)@self.B1@x1
        I0xp = torch.unsqueeze(self.I0@x0p,2) # torch.inverse(self.D1)@self.B1 is the projection matrix
        
        if self.k00 > 0 and self.k0p > 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,X00,I0xp,X0p),2)
        elif  self.k00 > 0 and self.k0p == 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0 = torch.cat((I0x, X00, I0xp ), 2)
        elif self.k00 > 0 and self.k0p > 0:
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0xp,X0p),2)
        else: 
            X0 = torch.cat((I0x,I0xp),2)

        '''order 1'''
        # x1n = self.D2@self.B1.T @torch.inverse(self.D1)@ x0
        # I1xn = torch.unsqueeze(self.I1@x1n,2)
        I1x = torch.unsqueeze(self.I1@x1,2)
        x1p = self.B2 @self.D3@ x2
        I1xp = torch.unsqueeze(self.I1@x1p,2)
        
        # if self.k1n > 0:
        #     X1nl = chebyshev(self.L1l, self.k1n, x1n)
        #     X1n = torch.cat((I1xn,X1nl),2)
        # else: 
        #     X1n = I1xn
        
        if self.k11>0 and self.k12>0:
            X1l = chebyshev(self.L1l, self.k11, x1)
            X1u = chebyshev(self.L1u, self.k12, x1)
            X11 = torch.cat((I1x, X1l, X1u),2)
        elif self.k11>0 and self.k12==0:
            X1l = chebyshev(self.L1l, self.k11, x1)
            X11 = torch.cat((I1x, X1l),2)  
        elif self.k11==0 and self.k12>0:
            X1u = chebyshev(self.L1u, self.k12, x1)
            X11 = torch.cat((I1x, X1u),2)
        else:
            X11 = I1x
            
        if self.k1p > 0:
            X1pu = chebyshev(self.L1u, self.k1p, x1p)
            X1p = torch.cat((I1xp, X1pu), 2)
        else:
            X1p = I1xp
            
        X1 = torch.cat((X11,X1p),2)
            
        '''order 2'''
        x2n = self.B2.T@torch.pinverse(self.D5)@x1
        I2xn = torch.unsqueeze(self.I2@x2n,2)
        I2x = torch.unsqueeze(self.I2@x2,2)
        if self.k2n>0 and self.k22 > 0:
            X2n = chebyshev(self.L2, self.k2n, x2n)
            X22 = chebyshev(self.L2, self.k22, x2)
            X2 = torch.cat((I2xn, X2n, I2x, X22), 2)
        elif self.k2n>0 and self.k22 == 0:
            X2n = chebyshev(self.L2, self.k2n, x2n)
            X2 = torch.cat((I2xn, X2n, I2x), 2)
        elif self.k2n==0 and self.k22 > 0:
            X22 = chebyshev(self.L2, self.k22, x2)
            X2 = torch.cat((I2xn, I2x, X22), 2)   
        else:
            X2 = torch.cat((I2xn,I2x),2)

        y0 = torch.einsum('nik,iok->no',X0,self.W0)
        y1 = torch.einsum('nik,iok->no',X1,self.W1)
        y2 = torch.einsum('nik,iok->no',X2,self.W2)
        y0 = self.sigma(y0)
        y1 = self.sigma(y1)
        y2 = self.sigma(y2)
        return y0,y1,y2
   
'''no x1 as input in x1: no edge to edge'''   
class sccnn_conv_no_e_to_e(nn.Module):
    def __init__(self, F_in, F_out, b1, b2, l0, l1l, l1u, l2, d1,d2,d3,d5, k00,k0p,k1n,k11,k12,k1p,k2n,k22, sigma):
        """
        F_in: number of input features per layer 
        F_out: number of output features per layer
        p: stands for positive, denoting the upper simplex oreder
        n: stands for negative, denoting the lower simplex order
        """
        super(sccnn_conv_no_e_to_e, self).__init__()
        self.F_in = F_in
        self.F_out = F_out 
        self.B1 = b1 
        self.B2 = b2
        self.L0 = l0
        self.L1l = l1l
        self.L1u = l1u        
        self.L2 = l2
        self.D1 = d1
        self.D2 = d2
        self.D3 = d3
        self.D5 = d5
        self.sigma = sigma
        self.k00 = k00
        self.k0p = k0p
        self.k1n = k1n 
        self.k11 = k11
        self.k12 = k12
        self.k1p = k1p
        self.k2n = k2n
        self.k22 = k22
        
        self.W0 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k00 + 1+self.k0p)))
        self.W1 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k1n + 1+self.k1p)))
        self.W2 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 1+self.k2n + 1+self.k22)))
        
        dim_0 = self.L0.size(dim=0)
        self.I0 = torch.eye(dim_0,device=self.L0.device)
        dim_1 = self.L1l.size(dim=0)
        self.I1 = torch.eye(dim_1,device=self.L1l.device)
        dim_2 = self.L2.size(dim=0)
        self.I2 = torch.eye(dim_2,device=self.L2.device)
        
        
        self.reset_parameters()
        print("created SCCNN layers")
    
    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W0.data, gain=gain)
        nn.init.xavier_uniform_(self.W1.data, gain=gain)
        nn.init.xavier_uniform_(self.W2.data, gain=gain)
        
    
    def forward(self,x_in):
        x0,x1,x2 = x_in
        
        '''order 0 '''
        I0x = torch.unsqueeze(self.I0@x0,2)
        x0p = torch.inverse(self.D1)@self.B1@x1
        I0xp = torch.unsqueeze(self.I0@x0p,2) # torch.inverse(self.D1)@self.B1 is the projection matrix
        
        if self.k00 > 0 and self.k0p > 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,X00,I0xp,X0p),2)
        elif  self.k00 > 0 and self.k0p == 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0 = torch.cat((I0x, X00 ), 2)
        elif self.k00 > 0 and self.k0p > 0:
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,I0xp,X0p),2)
        else: 
            X0 = torch.cat((I0x,I0xp),2)

        '''order 1'''
        x1n = self.D2@self.B1.T @torch.inverse(self.D1)@ x0
        I1xn = torch.unsqueeze(self.I1@x1n,2)
        # I1x = torch.unsqueeze(self.I1@x1,2)
        x1p = self.B2 @self.D3@ x2
        I1xp = torch.unsqueeze(self.I1@x1p,2)
        
        if self.k1n > 0:
            X1nl = chebyshev(self.L1l, self.k1n, x1n)
            X1n = torch.cat((I1xn,X1nl),2)
        else: 
            X1n = I1xn
        
        # if self.k11>0 and self.k12>0:
        #     X1l = chebyshev(self.L1l, self.k11, x1)
        #     X1u = chebyshev(self.L1u, self.k12, x1)
        #     X11 = torch.cat((I1x, X1l, X1u),2)
        # if self.k11>0 and self.k12==0:
        #     X1l = chebyshev(self.L1l, self.k11, x1)
        #     X11 = torch.cat((I1x, X1l),2)  
        # elif self.k11==0 and self.k12>0:
        #     X1u = chebyshev(self.L1u, self.k12, x1)
        #     X11 = torch.cat((I1x, X1u),2)
        # else:
        #     X11 = I1x
            
        if self.k1p > 0:
            X1pu = chebyshev(self.L1u, self.k1p, x1p)
            X1p = torch.cat((I1xp, X1pu), 2)
        else:
            X1p = I1xp
            
        X1 = torch.cat((X1n,X1p),2)
            
        '''order 2'''
        x2n = self.B2.T@torch.pinverse(self.D5)@x1
        I2xn = torch.unsqueeze(self.I2@x2n,2)
        I2x = torch.unsqueeze(self.I2@x2,2)
        if self.k2n>0 and self.k22 > 0:
            X2n = chebyshev(self.L2, self.k2n, x2n)
            X22 = chebyshev(self.L2, self.k22, x2)
            X2 = torch.cat((I2xn, X2n, I2x, X22), 2)
        elif self.k2n>0 and self.k22 == 0:
            X2n = chebyshev(self.L2, self.k2n, x2n)
            X2 = torch.cat((I2xn, X2n, I2x), 2)
        elif self.k2n==0 and self.k22 > 0:
            X22 = chebyshev(self.L2, self.k22, x2)
            X2 = torch.cat((I2x, X22), 2)   
        else:
            X2 = torch.cat((I2xn,I2x),2)

        y0 = torch.einsum('nik,iok->no',X0,self.W0)
        y1 = torch.einsum('nik,iok->no',X1,self.W1)
        y2 = torch.einsum('nik,iok->no',X2,self.W2)
        y0 = self.sigma(y0)
        y1 = self.sigma(y1)
        y2 = self.sigma(y2)
        return y0,y1,y2
   
'''no B2@x2 in x1: no triangle to edge'''    
class sccnn_conv_no_t_to_e(nn.Module):
    def __init__(self, F_in, F_out, b1, b2, l0, l1l, l1u, l2, d1,d2,d3,d5, k00,k0p,k1n,k11,k12,k1p,k2n,k22, sigma):
        """
        F_in: number of input features per layer 
        F_out: number of output features per layer
        p: stands for positive, denoting the upper simplex oreder
        n: stands for negative, denoting the lower simplex order
        """
        super(sccnn_conv_no_t_to_e, self).__init__()
        self.F_in = F_in
        self.F_out = F_out 
        self.B1 = b1 
        self.B2 = b2
        self.L0 = l0
        self.L1l = l1l
        self.L1u = l1u        
        self.L2 = l2
        self.D1 = d1
        self.D2 = d2
        self.D3 = d3
        self.D5 = d5
        self.sigma = sigma
        self.k00 = k00
        self.k0p = k0p
        self.k1n = k1n 
        self.k11 = k11
        self.k12 = k12
        self.k1p = k1p
        self.k2n = k2n
        self.k22 = k22
        
        self.W0 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k00 + 1+self.k0p)))
        self.W1 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k1n + 1+self.k11+self.k12)))
        #self.W2 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 1+self.k2n + 1+self.k22)))
        
        dim_0 = self.L0.size(dim=0)
        self.I0 = torch.eye(dim_0,device=self.L0.device)
        dim_1 = self.L1l.size(dim=0)
        self.I1 = torch.eye(dim_1,device=self.L1l.device)
        # dim_2 = self.L2.size(dim=0)
        # self.I2 = torch.eye(dim_2,device=self.L2.device)
        
        self.reset_parameters()
        print("created SCCNN layers but no triangles included")
    
    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W0.data, gain=gain)
        nn.init.xavier_uniform_(self.W1.data, gain=gain)
        #nn.init.xavier_uniform_(self.W2.data, gain=gain)
        
    
    def forward(self,x_in):
        x0,x1 = x_in
        
        '''order 0 '''
        I0x = torch.unsqueeze(self.I0@x0,2)
        x0p = torch.inverse(self.D1)@self.B1@x1
        I0xp = torch.unsqueeze(self.I0@x0p,2) # torch.inverse(self.D1)@self.B1 is the projection matrix
        
        if self.k00 > 0 and self.k0p > 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,X00,I0xp,X0p),2)
        elif  self.k00 > 0 and self.k0p == 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0 = torch.cat((I0x, X00, I0xp ), 2)
        elif self.k00 > 0 and self.k0p > 0:
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,I0xp,X0p),2)
        else: 
            X0 = torch.cat((I0x,I0xp),2)

        '''order 1'''
        x1n = self.D2@self.B1.T @torch.inverse(self.D1)@ x0
        I1xn = torch.unsqueeze(self.I1@x1n,2)
        I1x = torch.unsqueeze(self.I1@x1,2)
        # x1p = self.B2 @self.D3@ x2
        # I1xp = torch.unsqueeze(self.I1@x1p,2)
        
        if self.k1n > 0:
            X1nl = chebyshev(self.L1l, self.k1n, x1n)
            X1n = torch.cat((I1xn,X1nl),2)
        else: 
            X1n = I1xn
        
        if self.k11>0 and self.k12>0:
            X1l = chebyshev(self.L1l, self.k11, x1)
            X1u = chebyshev(self.L1u, self.k12, x1)
            X11 = torch.cat((I1x, X1l, X1u),2)
        elif self.k11>0 and self.k12==0:
            X1l = chebyshev(self.L1l, self.k11, x1)
            X11 = torch.cat((I1x, X1l),2)  
        # elif self.k11==0 and self.k12>0:
        #     X1u = chebyshev(self.L1u, self.k12, x1)
        #     X11 = torch.cat((I1x, X1u),2)
        else:
            X11 = I1x
            
        # if self.k1p > 0:
        #     X1pu = chebyshev(self.L1u, self.k1p, x1p)
        #     X1p = torch.cat((I1xp, X1pu), 2)
        # else:
        #     X1p = I1xp
            
        X1 = torch.cat((X1n,X11),2)
            
        # '''order 2'''
        # x2n = self.B2.T@torch.pinverse(self.D5)@x1
        # I2xn = torch.unsqueeze(self.I2@x2n,2)
        # I2x = torch.unsqueeze(self.I2@x2,2)
        # if self.k2n>0 and self.k22 > 0:
        #     X2n = chebyshev(self.L2, self.k2n, x2n)
        #     X22 = chebyshev(self.L2, self.k22, x2)
        #     X2 = torch.cat((I2xn, X2n, I2x, X22), 2)
        # elif self.k2n>0 and self.k22 == 0:
        #     X2n = chebyshev(self.L2, self.k2n, x2n)
        #     X2 = torch.cat((I2xn, X2n, I2x), 2)
        # elif self.k2n==0 and self.k22 > 0:
        #     X22 = chebyshev(self.L2, self.k22, x2)
        #     X2 = torch.cat((I2xn, I2x, X22), 2)   
        # else:
        #     X2 = torch.cat((I2xn,I2x),2)

        y0 = torch.einsum('nik,iok->no',X0,self.W0)
        y1 = torch.einsum('nik,iok->no',X1,self.W1)
        # y2 = torch.einsum('nik,iok->no',X2,self.W2)
        y0 = self.sigma(y0)
        y1 = self.sigma(y1)
        # y2 = self.sigma(y2)
        return y0,y1#,y2

'''no x0 in x0: no node to node'''
class sccnn_conv_no_n_to_n(nn.Module):
    def __init__(self, F_in, F_out, b1, b2, l0, l1l, l1u, l2, d1,d2,d3,d5, k00,k0p,k1n,k11,k12,k1p,k2n,k22, sigma):
        """
        F_in: number of input features per layer 
        F_out: number of output features per layer
        p: stands for positive, denoting the upper simplex oreder
        n: stands for negative, denoting the lower simplex order
        """
        super(sccnn_conv_no_n_to_n, self).__init__()
        self.F_in = F_in
        self.F_out = F_out 
        self.B1 = b1 
        self.B2 = b2
        self.L0 = l0
        self.L1l = l1l
        self.L1u = l1u        
        self.L2 = l2
        self.D1 = d1
        self.D2 = d2
        self.D3 = d3
        self.D5 = d5
        self.sigma = sigma
        self.k00 = k00
        self.k0p = k0p
        self.k1n = k1n 
        self.k11 = k11
        self.k12 = k12
        self.k1p = k1p
        self.k2n = k2n
        self.k22 = k22
        
        self.W0 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 1+self.k0p))) #only for projection from x1 to x0
        self.W1 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 1+self.k1n + 1+self.k11+self.k12 + 1+self.k1p)))
        self.W2 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 1+self.k2n + 1+self.k22)))
        
        dim_0 = self.L0.size(dim=0)
        self.I0 = torch.eye(dim_0,device=self.L0.device)
        dim_1 = self.L1l.size(dim=0)
        self.I1 = torch.eye(dim_1,device=self.L1l.device)
        dim_2 = self.L2.size(dim=0)
        self.I2 = torch.eye(dim_2,device=self.L2.device)
        
        
        self.reset_parameters()
        print("created SCCNN layers but no nodes included")
    
    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W0.data, gain=gain)
        nn.init.xavier_uniform_(self.W1.data, gain=gain)
        nn.init.xavier_uniform_(self.W2.data, gain=gain)
        
    
    def forward(self,x_in):
        x0,x1,x2 = x_in
        # '''order 0 '''
        #I0x = torch.unsqueeze(self.I0@x0,2)
        x0p = torch.inverse(self.D1)@self.B1@x1
        I0xp = torch.unsqueeze(self.I0@x0p,2) # torch.inverse(self.D1)@self.B1 is the projection matrix
        
        # if self.k00 > 0 and self.k0p > 0:
        #     X00 = chebyshev(self.L0, self.k00, x0)
        #     X0p = chebyshev(self.L0, self.k0p, x0p) 
        #     X0 = torch.cat((I0x,X00,I0xp,X0p),2)
        # elif  self.k00 > 0 and self.k0p == 0:
        #     X00 = chebyshev(self.L0, self.k00, x0)
        #     X0 = torch.cat((I0x, X00, I0xp ), 2)
        # elif self.k00 > 0 and self.k0p > 0:
        X0p = chebyshev(self.L0, self.k0p, x0p) 
        X0 = torch.cat((I0xp,X0p),2)
        # else: 
        #     X0 = torch.cat((I0x,I0xp),2)

        '''order 1'''
        x1n = self.D2@self.B1.T @torch.inverse(self.D1)@ x0
        I1xn = torch.unsqueeze(self.I1@x1n,2)
        I1x = torch.unsqueeze(self.I1@x1,2)
        x1p = self.B2 @self.D3@ x2
        I1xp = torch.unsqueeze(self.I1@x1p,2)
        
        if self.k1n > 0:
            X1nl = chebyshev(self.L1l, self.k1n, x1n)
            X1n = torch.cat((I1xn,X1nl),2)
        else: 
            X1n = I1xn
        
        if self.k11>0 and self.k12>0:
            X1l = chebyshev(self.L1l, self.k11, x1)
            X1u = chebyshev(self.L1u, self.k12, x1)
            X11 = torch.cat((I1x, X1l, X1u),2)
        elif self.k11>0 and self.k12==0:
            X1l = chebyshev(self.L1l, self.k11, x1)
            X11 = torch.cat((I1x, X1l),2)  
        elif self.k11==0 and self.k12>0:
            X1u = chebyshev(self.L1u, self.k12, x1)
            X11 = torch.cat((I1x, X1u),2)
        else:
            X11 = I1x
            
        if self.k1p > 0:
            X1pu = chebyshev(self.L1u, self.k1p, x1p)
            X1p = torch.cat((I1xp, X1pu), 2)
        else:
            X1p = I1xp
            
        X1 = torch.cat((X1n,X11,X1p),2)
            
        '''order 2'''
        x2n = self.B2.T@torch.pinverse(self.D5)@x1
        I2xn = torch.unsqueeze(self.I2@x2n,2)
        I2x = torch.unsqueeze(self.I2@x2,2)
        if self.k2n>0 and self.k22 > 0:
            X2n = chebyshev(self.L2, self.k2n, x2n)
            X22 = chebyshev(self.L2, self.k22, x2)
            X2 = torch.cat((I2xn, X2n, I2x, X22), 2)
        elif self.k2n>0 and self.k22 == 0:
            X2n = chebyshev(self.L2, self.k2n, x2n)
            X2 = torch.cat((I2xn, X2n, I2x), 2)
        elif self.k2n==0 and self.k22 > 0:
            X22 = chebyshev(self.L2, self.k22, x2)
            X2 = torch.cat((I2xn, I2x, X22), 2)   
        else:
            X2 = torch.cat((I2xn,I2x),2)

        y0 = torch.einsum('nik,iok->no',X0,self.W0)
        y1 = torch.einsum('nik,iok->no',X1,self.W1)
        y2 = torch.einsum('nik,iok->no',X2,self.W2)
        y0 = self.sigma(y0)
        y1 = self.sigma(y1)
        y2 = self.sigma(y2)
        return y0,y1,y2
  


'''no B1.T@x0 in x1, sc order 1: no node to edge'''
class sccnn_conv_no_n_to_e_sc_1(nn.Module):
    def __init__(self, F_in, F_out, b1, b2, l0, l1l, l1u, l2, d1,d2,d3,d5, k00,k0p,k1n,k11,k12,k1p,k2n,k22, sigma):
        """
        F_in: number of input features per layer 
        F_out: number of output features per layer
        p: stands for positive, denoting the upper simplex oreder
        n: stands for negative, denoting the lower simplex order
        """
        super(sccnn_conv_no_n_to_e_sc_1, self).__init__()
        self.F_in = F_in
        self.F_out = F_out 
        self.B1 = b1 
        self.B2 = b2
        self.L0 = l0
        self.L1l = l1l
        self.L1u = l1u        
        self.L2 = l2
        self.D1 = d1
        self.D2 = d2
        self.D3 = d3
        self.D5 = d5
        self.sigma = sigma
        self.k00 = k00
        self.k0p = k0p
        self.k1n = k1n 
        self.k11 = k11
        self.k12 = k12
        self.k1p = k1p
        self.k2n = k2n
        self.k22 = k22
        
        self.W0 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 1+self.k00 + 1+self.k0p))) #only for projection from x1 to x0
        self.W1 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 1+self.k11)))
        # self.W2 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 1+self.k2n + 1+self.k22)))
        
        dim_0 = self.L0.size(dim=0)
        self.I0 = torch.eye(dim_0,device=self.L0.device)
        dim_1 = self.L1l.size(dim=0)
        self.I1 = torch.eye(dim_1,device=self.L1l.device)
        dim_2 = self.L2.size(dim=0)
        self.I2 = torch.eye(dim_2,device=self.L2.device)
        
        
        self.reset_parameters()
        print("created SCCNN layers but no nodes included")
    
    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W0.data, gain=gain)
        nn.init.xavier_uniform_(self.W1.data, gain=gain)
        # nn.init.xavier_uniform_(self.W2.data, gain=gain)
        
    
    def forward(self,x_in):
        x0,x1 = x_in
        # '''order 0 '''
        I0x = torch.unsqueeze(self.I0@x0,2)
        x0p = torch.inverse(self.D1)@self.B1@x1
        I0xp = torch.unsqueeze(self.I0@x0p,2) # torch.inverse(self.D1)@self.B1 is the projection matrix
        
        if self.k00 > 0 and self.k0p > 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,X00,I0xp,X0p),2)
        elif  self.k00 > 0 and self.k0p == 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0 = torch.cat((I0x, X00, I0xp ), 2)
        elif self.k00 > 0 and self.k0p > 0:
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0xp,X0p),2)
        else: 
            X0 = torch.cat((I0x,I0xp),2)

        '''order 1'''
        # x1n = self.D2@self.B1.T @torch.inverse(self.D1)@ x0
        # I1xn = torch.unsqueeze(self.I1@x1n,2)
        I1x = torch.unsqueeze(self.I1@x1,2)
        # x1p = self.B2 @self.D3@ x2
        # I1xp = torch.unsqueeze(self.I1@x1p,2)
        
        # if self.k1n > 0:
        #     X1nl = chebyshev(self.L1l, self.k1n, x1n)
        #     X1n = torch.cat((I1xn,X1nl),2)
        # else: 
        #     X1n = I1xn
        
        if self.k11>0 and self.k12>0:
            X1l = chebyshev(self.L1l, self.k11, x1)
            # X1u = chebyshev(self.L1u, self.k12, x1)
            X11 = torch.cat((I1x, X1l),2)
        elif self.k11>0 and self.k12==0:
            X1l = chebyshev(self.L1l, self.k11, x1)
            X11 = torch.cat((I1x, X1l),2)  
        # elif self.k11==0 and self.k12>0:
        #     X1u = chebyshev(self.L1u, self.k12, x1)
        #     X11 = torch.cat((I1x, X1u),2)
        else:
            X11 = I1x
            
        # if self.k1p > 0:
        #     X1pu = chebyshev(self.L1u, self.k1p, x1p)
        #     X1p = torch.cat((I1xp, X1pu), 2)
        # else:
        #     X1p = I1xp
            
        # X1 = torch.cat((X11,X1p),2)
            
        # '''order 2'''
        # x2n = self.B2.T@torch.pinverse(self.D5)@x1
        # I2xn = torch.unsqueeze(self.I2@x2n,2)
        # I2x = torch.unsqueeze(self.I2@x2,2)
        # if self.k2n>0 and self.k22 > 0:
        #     X2n = chebyshev(self.L2, self.k2n, x2n)
        #     X22 = chebyshev(self.L2, self.k22, x2)
        #     X2 = torch.cat((I2xn, X2n, I2x, X22), 2)
        # elif self.k2n>0 and self.k22 == 0:
        #     X2n = chebyshev(self.L2, self.k2n, x2n)
        #     X2 = torch.cat((I2xn, X2n, I2x), 2)
        # elif self.k2n==0 and self.k22 > 0:
        #     X22 = chebyshev(self.L2, self.k22, x2)
        #     X2 = torch.cat((I2xn, I2x, X22), 2)   
        # else:
        #     X2 = torch.cat((I2xn,I2x),2)

        y0 = torch.einsum('nik,iok->no',X0,self.W0)
        y1 = torch.einsum('nik,iok->no',X11,self.W1)
        # y2 = torch.einsum('nik,iok->no',X2,self.W2)
        y0 = self.sigma(y0)
        y1 = self.sigma(y1)
        # y2 = self.sigma(y2)
        return y0,y1 #,y2

'''no x0 in x0, sc order 1: no node to node'''
class sccnn_conv_no_n_to_n_sc_1(nn.Module):
    def __init__(self, F_in, F_out, b1, b2, l0, l1l, l1u, l2, d1,d2,d3,d5, k00,k0p,k1n,k11,k12,k1p,k2n,k22, sigma):
        """
        F_in: number of input features per layer 
        F_out: number of output features per layer
        p: stands for positive, denoting the upper simplex oreder
        n: stands for negative, denoting the lower simplex order
        """
        super(sccnn_conv_no_n_to_n_sc_1, self).__init__()
        self.F_in = F_in
        self.F_out = F_out 
        self.B1 = b1 
        self.B2 = b2
        self.L0 = l0
        self.L1l = l1l
        self.L1u = l1u        
        self.L2 = l2
        self.D1 = d1
        self.D2 = d2
        self.D3 = d3
        self.D5 = d5
        self.sigma = sigma
        self.k00 = k00
        self.k0p = k0p
        self.k1n = k1n 
        self.k11 = k11
        self.k12 = k12
        self.k1p = k1p
        self.k2n = k2n
        self.k22 = k22
        
        self.W0 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 1+self.k0p))) #only for projection from x1 to x0
        self.W1 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 1+self.k1n + 1+self.k11)))
        #self.W2 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 1+self.k2n + 1+self.k22)))
        
        dim_0 = self.L0.size(dim=0)
        self.I0 = torch.eye(dim_0,device=self.L0.device)
        dim_1 = self.L1l.size(dim=0)
        self.I1 = torch.eye(dim_1,device=self.L1l.device)
        dim_2 = self.L2.size(dim=0)
        self.I2 = torch.eye(dim_2,device=self.L2.device)
        
        
        self.reset_parameters()
        print("created SCCNN layers but no nodes included")
    
    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W0.data, gain=gain)
        nn.init.xavier_uniform_(self.W1.data, gain=gain)
        #nn.init.xavier_uniform_(self.W2.data, gain=gain)
        
    
    def forward(self,x_in):
        x0,x1 = x_in
        # '''order 0 '''
        #I0x = torch.unsqueeze(self.I0@x0,2)
        x0p = torch.inverse(self.D1)@self.B1@x1
        I0xp = torch.unsqueeze(self.I0@x0p,2) # torch.inverse(self.D1)@self.B1 is the projection matrix
        
        # if self.k00 > 0 and self.k0p > 0:
        #     X00 = chebyshev(self.L0, self.k00, x0)
        #     X0p = chebyshev(self.L0, self.k0p, x0p) 
        #     X0 = torch.cat((I0x,X00,I0xp,X0p),2)
        # elif  self.k00 > 0 and self.k0p == 0:
        #     X00 = chebyshev(self.L0, self.k00, x0)
        #     X0 = torch.cat((I0x, X00, I0xp ), 2)
        # elif self.k00 > 0 and self.k0p > 0:
        X0p = chebyshev(self.L0, self.k0p, x0p) 
        X0 = torch.cat((I0xp,X0p),2)
        # else: 
        #     X0 = torch.cat((I0x,I0xp),2)

        '''order 1'''
        x1n = self.D2@self.B1.T @torch.inverse(self.D1)@ x0
        I1xn = torch.unsqueeze(self.I1@x1n,2)
        I1x = torch.unsqueeze(self.I1@x1,2)
        # x1p = self.B2 @self.D3@ x2
        # I1xp = torch.unsqueeze(self.I1@x1p,2)
        
        if self.k1n > 0:
            X1nl = chebyshev(self.L1l, self.k1n, x1n)
            X1n = torch.cat((I1xn,X1nl),2)
        else: 
            X1n = I1xn
        
        if self.k11>0 and self.k12>0:
            X1l = chebyshev(self.L1l, self.k11, x1)
            #X1u = chebyshev(self.L1u, self.k12, x1)
            X11 = torch.cat((I1x, X1l),2)
        elif self.k11>0 and self.k12==0:
            X1l = chebyshev(self.L1l, self.k11, x1)
            X11 = torch.cat((I1x, X1l),2)  
        # elif self.k11==0 and self.k12>0:
        #     X1u = chebyshev(self.L1u, self.k12, x1)
        #     X11 = torch.cat((I1x, X1u),2)
        else:
            X11 = I1x
            
        # if self.k1p > 0:
        #     X1pu = chebyshev(self.L1u, self.k1p, x1p)
        #     X1p = torch.cat((I1xp, X1pu), 2)
        # else:
        #     X1p = I1xp
            
        X1 = torch.cat((X1n,X11),2)
            
        # '''order 2'''
        # x2n = self.B2.T@torch.pinverse(self.D5)@x1
        # I2xn = torch.unsqueeze(self.I2@x2n,2)
        # I2x = torch.unsqueeze(self.I2@x2,2)
        # if self.k2n>0 and self.k22 > 0:
        #     X2n = chebyshev(self.L2, self.k2n, x2n)
        #     X22 = chebyshev(self.L2, self.k22, x2)
        #     X2 = torch.cat((I2xn, X2n, I2x, X22), 2)
        # elif self.k2n>0 and self.k22 == 0:
        #     X2n = chebyshev(self.L2, self.k2n, x2n)
        #     X2 = torch.cat((I2xn, X2n, I2x), 2)
        # elif self.k2n==0 and self.k22 > 0:
        #     X22 = chebyshev(self.L2, self.k22, x2)
        #     X2 = torch.cat((I2xn, I2x, X22), 2)   
        # else:
        #     X2 = torch.cat((I2xn,I2x),2)

        y0 = torch.einsum('nik,iok->no',X0,self.W0)
        y1 = torch.einsum('nik,iok->no',X1,self.W1)
        # y2 = torch.einsum('nik,iok->no',X2,self.W2)
        y0 = self.sigma(y0)
        y1 = self.sigma(y1)
        # y2 = self.sigma(y2)
        return y0,y1# ,y2
   
'''no x1 as input in x1, sc order 1: no edge to edge'''   
class sccnn_conv_no_e_to_e_sc_1(nn.Module):
    def __init__(self, F_in, F_out, b1, b2, l0, l1l, l1u, l2, d1,d2,d3,d5, k00,k0p,k1n,k11,k12,k1p,k2n,k22, sigma):
        """
        F_in: number of input features per layer 
        F_out: number of output features per layer
        p: stands for positive, denoting the upper simplex oreder
        n: stands for negative, denoting the lower simplex order
        """
        super(sccnn_conv_no_e_to_e_sc_1, self).__init__()
        self.F_in = F_in
        self.F_out = F_out 
        self.B1 = b1 
        self.B2 = b2
        self.L0 = l0
        self.L1l = l1l
        self.L1u = l1u        
        self.L2 = l2
        self.D1 = d1
        self.D2 = d2
        self.D3 = d3
        self.D5 = d5
        self.sigma = sigma
        self.k00 = k00
        self.k0p = k0p
        self.k1n = k1n 
        self.k11 = k11
        self.k12 = k12
        self.k1p = k1p
        self.k2n = k2n
        self.k22 = k22
        
        self.W0 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k00 + 1+self.k0p)))
        self.W1 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k1n )))
        # self.W2 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 1+self.k2n + 1+self.k22)))
        
        dim_0 = self.L0.size(dim=0)
        self.I0 = torch.eye(dim_0,device=self.L0.device)
        dim_1 = self.L1l.size(dim=0)
        self.I1 = torch.eye(dim_1,device=self.L1l.device)
        dim_2 = self.L2.size(dim=0)
        self.I2 = torch.eye(dim_2,device=self.L2.device)
        
        
        self.reset_parameters()
        print("created SCCNN layers")
    
    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W0.data, gain=gain)
        nn.init.xavier_uniform_(self.W1.data, gain=gain)
        # nn.init.xavier_uniform_(self.W2.data, gain=gain)
        
    
    def forward(self,x_in):
        x0,x1 = x_in
        
        '''order 0 '''
        I0x = torch.unsqueeze(self.I0@x0,2)
        x0p = torch.inverse(self.D1)@self.B1@x1
        I0xp = torch.unsqueeze(self.I0@x0p,2) # torch.inverse(self.D1)@self.B1 is the projection matrix
        
        if self.k00 > 0 and self.k0p > 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,X00,I0xp,X0p),2)
        elif  self.k00 > 0 and self.k0p == 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0 = torch.cat((I0x, X00 ), 2)
        elif self.k00 > 0 and self.k0p > 0:
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,I0xp,X0p),2)
        else: 
            X0 = torch.cat((I0x,I0xp),2)

        '''order 1'''
        x1n = self.D2@self.B1.T @torch.inverse(self.D1)@ x0
        I1xn = torch.unsqueeze(self.I1@x1n,2)
        # I1x = torch.unsqueeze(self.I1@x1,2)
        # x1p = self.B2 @self.D3@ x2
        # I1xp = torch.unsqueeze(self.I1@x1p,2)
        
        if self.k1n > 0:
            X1nl = chebyshev(self.L1l, self.k1n, x1n)
            X1n = torch.cat((I1xn,X1nl),2)
        else: 
            X1n = I1xn
        
        # if self.k11>0 and self.k12>0:
        #     X1l = chebyshev(self.L1l, self.k11, x1)
        #     X1u = chebyshev(self.L1u, self.k12, x1)
        #     X11 = torch.cat((I1x, X1l, X1u),2)
        # if self.k11>0 and self.k12==0:
        #     X1l = chebyshev(self.L1l, self.k11, x1)
        #     X11 = torch.cat((I1x, X1l),2)  
        # elif self.k11==0 and self.k12>0:
        #     X1u = chebyshev(self.L1u, self.k12, x1)
        #     X11 = torch.cat((I1x, X1u),2)
        # else:
        #     X11 = I1x
            
        # if self.k1p > 0:
        #     X1pu = chebyshev(self.L1u, self.k1p, x1p)
        #     X1p = torch.cat((I1xp, X1pu), 2)
        # else:
        #     X1p = I1xp
            
        # X1 = torch.cat((X1n,X1p),2)
            
        # '''order 2'''
        # x2n = self.B2.T@torch.pinverse(self.D5)@x1
        # I2xn = torch.unsqueeze(self.I2@x2n,2)
        # I2x = torch.unsqueeze(self.I2@x2,2)
        # if self.k2n>0 and self.k22 > 0:
        #     X2n = chebyshev(self.L2, self.k2n, x2n)
        #     X22 = chebyshev(self.L2, self.k22, x2)
        #     X2 = torch.cat((I2xn, X2n, I2x, X22), 2)
        # elif self.k2n>0 and self.k22 == 0:
        #     X2n = chebyshev(self.L2, self.k2n, x2n)
        #     X2 = torch.cat((I2xn, X2n, I2x), 2)
        # elif self.k2n==0 and self.k22 > 0:
        #     X22 = chebyshev(self.L2, self.k22, x2)
        #     X2 = torch.cat((I2x, X22), 2)   
        # else:
        #     X2 = torch.cat((I2xn,I2x),2)

        y0 = torch.einsum('nik,iok->no',X0,self.W0)
        y1 = torch.einsum('nik,iok->no',X1n,self.W1)
        # y2 = torch.einsum('nik,iok->no',X2,self.W2)
        y0 = self.sigma(y0)
        y1 = self.sigma(y1)
        # y2 = self.sigma(y2)
        return y0,y1#,y2
 
'''no B1@x1 as input in x0, sc order 1: no edge to node, i.e., gnn''' 
class sccnn_conv_no_e_to_n_sc_1(nn.Module):
    def __init__(self, F_in, F_out, b1, b2, l0, l1l, l1u, l2, d1,d2,d3,d5, k00,k0p,k1n,k11,k12,k1p,k2n,k22, sigma):
        """
        F_in: number of input features per layer 
        F_out: number of output features per layer
        p: stands for positive, denoting the upper simplex oreder
        n: stands for negative, denoting the lower simplex order
        """
        super(sccnn_conv_no_e_to_n_sc_1, self).__init__()
        self.F_in = F_in
        self.F_out = F_out 
        self.B1 = b1 
        self.B2 = b2
        self.L0 = l0
        self.L1l = l1l
        self.L1u = l1u        
        self.L2 = l2
        self.D1 = d1
        self.D2 = d2
        self.D3 = d3
        self.D5 = d5
        self.sigma = sigma
        self.k00 = k00
        self.k0p = k0p
        self.k1n = k1n 
        self.k11 = k11
        self.k12 = k12
        self.k1p = k1p
        self.k2n = k2n
        self.k22 = k22
        
        self.W0 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k00)))
        self.W1 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k1n + 1+self.k11)))
        # self.W2 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 1+self.k2n + 1+self.k22)))
        
        dim_0 = self.L0.size(dim=0)
        self.I0 = torch.eye(dim_0,device=self.L0.device)
        dim_1 = self.L1l.size(dim=0)
        self.I1 = torch.eye(dim_1,device=self.L1l.device)
        dim_2 = self.L2.size(dim=0)
        self.I2 = torch.eye(dim_2,device=self.L2.device)
        
        
        self.reset_parameters()
        print("created SCCNN layers")
    
    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W0.data, gain=gain)
        nn.init.xavier_uniform_(self.W1.data, gain=gain)
        # nn.init.xavier_uniform_(self.W2.data, gain=gain)
        
    
    def forward(self,x_in):
        x0,x1 = x_in
        
        '''order 0 '''
        I0x = torch.unsqueeze(self.I0@x0,2)
        # x0p = torch.inverse(self.D1)@self.B1@x1
        # I0xp = torch.unsqueeze(self.I0@x0p,2) # torch.inverse(self.D1)@self.B1 is the projection matrix
        
        if self.k00 > 0 and self.k0p > 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            # X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,X00),2)
        # elif  self.k00 > 0 and self.k0p == 0:
        #     X00 = chebyshev(self.L0, self.k00, x0)
        #     X0 = torch.cat((I0x, X00 ), 2)
        # elif self.k00 > 0 and self.k0p > 0:
        #     X0p = chebyshev(self.L0, self.k0p, x0p) 
        #     X0 = torch.cat((I0x,I0xp,X0p),2)
        # else: 
        #     X0 = torch.cat((I0x,I0xp),2)

        '''order 1'''
        x1n = self.D2@self.B1.T @torch.inverse(self.D1)@ x0
        I1xn = torch.unsqueeze(self.I1@x1n,2)
        I1x = torch.unsqueeze(self.I1@x1,2)
        # x1p = self.B2 @self.D3@ x2
        # I1xp = torch.unsqueeze(self.I1@x1p,2)
        
        if self.k1n > 0:
            X1nl = chebyshev(self.L1l, self.k1n, x1n)
            X1n = torch.cat((I1xn,X1nl),2)
        else: 
            X1n = I1xn
        
        if self.k11>0 and self.k12>0:
            X1l = chebyshev(self.L1l, self.k11, x1)
            #X1u = chebyshev(self.L1u, self.k12, x1)
            X11 = torch.cat((I1x, X1l),2)
        elif self.k11>0 and self.k12==0:
            X1l = chebyshev(self.L1l, self.k11, x1)
            X11 = torch.cat((I1x, X1l),2)  
        # elif self.k11==0 and self.k12>0:
        #     X1u = chebyshev(self.L1u, self.k12, x1)
        #     X11 = torch.cat((I1x, X1u),2)
        else:
            X11 = I1x
            
        # if self.k1p > 0:
        #     X1pu = chebyshev(self.L1u, self.k1p, x1p)
        #     X1p = torch.cat((I1xp, X1pu), 2)
        # else:
        #     X1p = I1xp
            
        X1 = torch.cat((X1n,X11),2)
            
        # '''order 2'''
        # x2n = self.B2.T@torch.pinverse(self.D5)@x1
        # I2xn = torch.unsqueeze(self.I2@x2n,2)
        # I2x = torch.unsqueeze(self.I2@x2,2)
        # if self.k2n>0 and self.k22 > 0:
        #     X2n = chebyshev(self.L2, self.k2n, x2n)
        #     X22 = chebyshev(self.L2, self.k22, x2)
        #     X2 = torch.cat((I2xn, X2n, I2x, X22), 2)
        # elif self.k2n>0 and self.k22 == 0:
        #     X2n = chebyshev(self.L2, self.k2n, x2n)
        #     X2 = torch.cat((I2xn, X2n, I2x), 2)
        # elif self.k2n==0 and self.k22 > 0:
        #     X22 = chebyshev(self.L2, self.k22, x2)
        #     X2 = torch.cat((I2x, X22), 2)   
        # else:
        #     X2 = torch.cat((I2xn,I2x),2)

        y0 = torch.einsum('nik,iok->no',X0,self.W0)
        y1 = torch.einsum('nik,iok->no',X1,self.W1)
        # y2 = torch.einsum('nik,iok->no',X2,self.W2)
        y0 = self.sigma(y0)
        y1 = self.sigma(y1)
        # y2 = self.sigma(y2)
        return y0,y1#,y2
 
'''sccnn stability where a D4 matrix (identity) was added for the purpose of adding perturbations to it'''
class sccnn_conv_stability(nn.Module):
    def __init__(self, F_in, F_out, b1, b2, l0, l1l, l1u, l2, d1,d2,d3,d4,d5, k00,k0p,k1n,k11,k12,k1p,k2n,k22, sigma):
        """
        F_in: number of input features per layer 
        F_out: number of output features per layer
        p: stands for positive, denoting the upper simplex oreder
        n: stands for negative, denoting the lower simplex order
        """
        super(sccnn_conv_stability, self).__init__()
        self.F_in = F_in
        self.F_out = F_out 
        self.B1 = b1 
        self.B2 = b2
        self.L0 = l0
        self.L1l = l1l
        self.L1u = l1u        
        self.L2 = l2
        self.D1 = d1
        self.D2 = d2
        self.D3 = d3
        self.D4 = d4
        self.D5 = d5
        self.sigma = sigma
        self.k00 = k00
        self.k0p = k0p
        self.k1n = k1n 
        self.k11 = k11
        self.k12 = k12
        self.k1p = k1p
        self.k2n = k2n
        self.k22 = k22
        
        self.W0 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k00 + 1+self.k0p)))
        self.W1 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k1n + 1+self.k11+self.k12 + 1+self.k1p)))
        self.W2 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 1+self.k2n + 1+self.k22)))
        
        dim_0 = self.L0.size(dim=0)
        self.I0 = torch.eye(dim_0,device=self.L0.device)
        dim_1 = self.L1l.size(dim=0)
        self.I1 = torch.eye(dim_1,device=self.L1l.device)
        dim_2 = self.L2.size(dim=0)
        self.I2 = torch.eye(dim_2,device=self.L2.device)
        
        
        self.reset_parameters()
        print("created SCCNN layers")
    
    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W0.data, gain=gain)
        nn.init.xavier_uniform_(self.W1.data, gain=gain)
        nn.init.xavier_uniform_(self.W2.data, gain=gain)
        
    
    def forward(self,x_in):
        x0,x1,x2 = x_in
        
        '''order 0 '''
        I0x = torch.unsqueeze(self.I0@x0,2)
        x0p = torch.inverse(self.D1)@self.B1@x1
        I0xp = torch.unsqueeze(self.I0@x0p,2) # torch.inverse(self.D1)@self.B1 is the projection matrix
        
        if self.k00 > 0 and self.k0p > 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,X00,I0xp,X0p),2)
        elif  self.k00 > 0 and self.k0p == 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0 = torch.cat((I0x, X00, I0xp ), 2)
        elif self.k00 > 0 and self.k0p > 0:
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,I0xp,X0p),2)
        else: 
            X0 = torch.cat((I0x,I0xp),2)

        '''order 1'''
        x1n = self.D2@self.B1.T @torch.inverse(self.D1)@ x0
        I1xn = torch.unsqueeze(self.I1@x1n,2)
        I1x = torch.unsqueeze(self.I1@x1,2)
        x1p = self.B2 @self.D3@ x2
        I1xp = torch.unsqueeze(self.I1@x1p,2)
        
        if self.k1n > 0:
            X1nl = chebyshev(self.L1l, self.k1n, x1n)
            X1n = torch.cat((I1xn,X1nl),2)
        else: 
            X1n = I1xn
        
        if self.k11>0 and self.k12>0:
            X1l = chebyshev(self.L1l, self.k11, x1)
            X1u = chebyshev(self.L1u, self.k12, x1)
            X11 = torch.cat((I1x, X1l, X1u),2)
        elif self.k11>0 and self.k12==0:
            X1l = chebyshev(self.L1l, self.k11, x1)
            X11 = torch.cat((I1x, X1l),2)  
        elif self.k11==0 and self.k12>0:
            X1u = chebyshev(self.L1u, self.k12, x1)
            X11 = torch.cat((I1x, X1u),2)
        else:
            X11 = I1x
            
        if self.k1p > 0:
            X1pu = chebyshev(self.L1u, self.k1p, x1p)
            X1p = torch.cat((I1xp, X1pu), 2)
        else:
            X1p = I1xp
            
        X1 = torch.cat((X1n,X11,X1p),2)
            
        '''order 2'''
        x2n = self.D4@self.B2.T@torch.pinverse(self.D5)@x1
        I2xn = torch.unsqueeze(self.I2@x2n,2)
        I2x = torch.unsqueeze(self.I2@x2,2)
        if self.k2n>0 and self.k22 > 0:
            X2n = chebyshev(self.L2, self.k2n, x2n)
            X22 = chebyshev(self.L2, self.k22, x2)
            X2 = torch.cat((I2xn, X2n, I2x, X22), 2)
        elif self.k2n>0 and self.k22 == 0:
            X2n = chebyshev(self.L2, self.k2n, x2n)
            X2 = torch.cat((I2xn, X2n, I2x), 2)
        elif self.k2n==0 and self.k22 > 0:
            X22 = chebyshev(self.L2, self.k22, x2)
            X2 = torch.cat((I2xn, I2x, X22), 2)   
        else:
            X2 = torch.cat((I2xn,I2x),2)

        y0 = torch.einsum('nik,iok->no',X0,self.W0)
        y1 = torch.einsum('nik,iok->no',X1,self.W1)
        y2 = torch.einsum('nik,iok->no',X2,self.W2)
        y0 = self.sigma(y0)
        y1 = self.sigma(y1)
        y2 = self.sigma(y2)
        return y0,y1,y2

