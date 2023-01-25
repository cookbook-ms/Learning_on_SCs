#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch 
import torch.nn as nn
import torch.nn.functional as F
from chebyshev import chebyshev

class sccnn_conv(nn.Module):
    def __init__(self, F_in, F_out, b1, b2, b3, l0, l1l, l1u, l2l, l2u, l3, d1,d2,d3,d5, d10, d20, d30, k00, k0p,k1n,k11,k12,k1p, k2n,k21,k22,k2p,k3n,k33, sigma):
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
        self.B3 = b3
        self.L0 = l0 
        self.L1l = l1l
        self.L1u = l1u
        self.L2l = l2l
        self.L2u = l2u        
        self.L3 = l3
        self.D1 = d1
        self.D2 = d2
        self.D3 = d3
        self.D5 = d5
        self.D10 = d10
        self.D20 = d20
        self.D30 = d30
        self.sigma = sigma

        self.k00 = k00
        self.k0p = k0p
        
        self.k1n = k1n 
        self.k11 = k11
        self.k12 = k12
        self.k1p = k1p
        
        self.k2n = k2n
        self.k21 = k21
        self.k22 = k22
        self.k2p = k2p
        
        self.k3n = k3n
        self.k33 = k33
        
        self.W0 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k00 + 1+self.k0p)))
        self.W1 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k1n + 1+self.k11+self.k12 + 1+self.k1p)))
        self.W2 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k2n + 1+self.k21+self.k22 + 1+self.k2p)))
        self.W3 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 1+self.k3n + 1+self.k33)))
        
        dim_0 = self.L0.size(dim=0)
        self.I0 = torch.eye(dim_0,device=self.L0.device)
        dim_1 = self.L1l.size(dim=0)
        self.I1 = torch.eye(dim_1,device=self.L1l.device)
        dim_2 = self.L2l.size(dim=0)
        self.I2 = torch.eye(dim_2,device=self.L2l.device)
        dim_3 = self.L3.size(dim=0)
        self.I3 = torch.eye(dim_3,device=self.L3.device)
        
        self.reset_parameters()
        print("created SCCNN layers")
    
    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W0.data, gain=gain)
        nn.init.xavier_uniform_(self.W1.data, gain=gain)
        nn.init.xavier_uniform_(self.W2.data, gain=gain)
        nn.init.xavier_uniform_(self.W3.data, gain=gain)
        
        
    
    def forward(self,x_in):
        '''note that x0 is useless since it will be removed by boundary identity property'''
        x0,x1,x2,x3 = x_in
        
        '''order 0'''
        I0x = torch.unsqueeze(self.I0@x0,2)
        x0p = torch.inverse(self.D10)@self.B1@x1
        I0xp = torch.unsqueeze(self.I0@x0p,2) # torch.inverse(self.D1)@self.B1 is the projection matrix

        if self.k00 > 0 and self.k0p > 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,X00,I0xp,X0p),2)
        elif  self.k00 > 0 and self.k0p == 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0 = torch.cat((I0x, X00, I0xp ), 2)
        elif self.k00 == 0 and self.k0p > 0:
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,I0xp,X0p),2)
        else: 
            X0 = torch.cat((I0x,I0xp),2)
            
        '''order 1'''
        x1n = self.D20@self.B1.T @torch.inverse(self.D10)@ x0
        I1xn = torch.unsqueeze(self.I1@x1n,2)
        I1x = torch.unsqueeze(self.I1@x1,2)
        x1p = torch.pinverse(self.D1)@self.B2@x2
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
        x2n = self.D2@self.B2.T @torch.pinverse(self.D1)@ x1
        I2xn = torch.unsqueeze(self.I2@x2n,2)
        I2x = torch.unsqueeze(self.I2@x2,2)
        x2p = self.B3@self.D3@x3
        I2xp = torch.unsqueeze(self.I2@x2p,2)
        
        if self.k2n > 0:
            X2nl = chebyshev(self.L2l, self.k2n, x2n)
            X2n = torch.cat((I2xn,X2nl),2)
        else: 
            X2n = I2xn
        
        if self.k21>0 and self.k22>0:
            X2l = chebyshev(self.L2l, self.k21, x2)
            X2u = chebyshev(self.L2u, self.k22, x2)
            X22 = torch.cat((I2x, X2l, X2u),2)
        elif self.k21>0 and self.k22==0:
            X2l = chebyshev(self.L2l, self.k21, x2)
            X22 = torch.cat((I2x, X2l),2)  
        elif self.k21==0 and self.k22>0:
            X2u = chebyshev(self.L2u, self.k22, x2)
            X22 = torch.cat((I2x, X2u),2)
        else:
            X22 = I2x
            
        if self.k2p > 0:
            X2pu = chebyshev(self.L2u, self.k2p, x2p)
            X2p = torch.cat((I2xp, X2pu), 2)
        else:
            X2p = I2xp
            
        X2 = torch.cat((X2n,X22,X2p),2)
            
        '''order 3'''
        x3n = self.B3.T@torch.pinverse(self.D5)@x2
        I3xn = torch.unsqueeze(self.I3@x3n,2)
        I3x = torch.unsqueeze(self.I3@x3,2)
        if self.k3n>0 and self.k33 > 0:
            X3n = chebyshev(self.L3, self.k3n, x3n)
            X33 = chebyshev(self.L3, self.k33, x3)
            X3 = torch.cat((I3xn, X3n, I3x, X33), 2)
        elif self.k3n>0 and self.k33 == 0:
            X3n = chebyshev(self.L3, self.k3n, x3n)
            X3 = torch.cat((I3xn, X3n, I3x), 2)
        elif self.k3n==0 and self.k33 > 0:
            X33 = chebyshev(self.L3, self.k33, x2)
            X3 = torch.cat((I3xn, I3x, X33), 2)   
        else:
            X3 = torch.cat((I3xn,I3x),2)

        y0 = torch.einsum('nik,iok->no',X0,self.W0)
        y1 = torch.einsum('nik,iok->no',X1,self.W1)
        y2 = torch.einsum('nik,iok->no',X2,self.W2)
        y3 = torch.einsum('nik,iok->no',X3,self.W3)
        y0 = self.sigma(y0)
        y1 = self.sigma(y1)
        y2 = self.sigma(y2)
        y3 = self.sigma(y3)
        return y0,y1,y2,y3
    
    
class sccnn_conv_id(nn.Module):
    def __init__(self, F_in, F_out, b1, b2, b3, l0, l1l, l1u, l2l, l2u, l3, d1,d2,d3,d5, d10, d20, d30, k00, k0p,k1n,k11,k12,k1p, k2n,k21,k22,k2p,k3n,k33, sigma):
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
        self.B3 = b3
        self.L0 = l0 
        self.L1l = l1l
        self.L1u = l1u
        self.L2l = l2l
        self.L2u = l2u        
        self.L3 = l3
        self.D1 = d1
        self.D2 = d2
        self.D3 = d3
        self.D5 = d5
        self.D10 = d10
        self.D20 = d20
        self.D30 = d30
        self.sigma = sigma

        self.k00 = k00
        self.k0p = k0p
        
        self.k1n = k1n 
        self.k11 = k11
        self.k12 = k12
        self.k1p = k1p
        
        self.k2n = k2n
        self.k21 = k21
        self.k22 = k22
        self.k2p = k2p
        
        self.k3n = k3n
        self.k33 = k33
        
        self.W0 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k00 + 1+self.k0p)))
        self.W1 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k1n + 1+self.k11+self.k12 + 1+self.k1p)))
        self.W2 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k2n + 1+self.k21+self.k22 + 1+self.k2p)))
        self.W3 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 1+self.k3n + 1+self.k33)))
        
        dim_0 = self.L0.size(dim=0)
        self.I0 = torch.eye(dim_0,device=self.L0.device)
        dim_1 = self.L1l.size(dim=0)
        self.I1 = torch.eye(dim_1,device=self.L1l.device)
        dim_2 = self.L2l.size(dim=0)
        self.I2 = torch.eye(dim_2,device=self.L2l.device)
        dim_3 = self.L3.size(dim=0)
        self.I3 = torch.eye(dim_3,device=self.L3.device)
        
        self.reset_parameters()
        print("created SCCNN layers")
    
    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W0.data, gain=gain)
        nn.init.xavier_uniform_(self.W1.data, gain=gain)
        nn.init.xavier_uniform_(self.W2.data, gain=gain)
        nn.init.xavier_uniform_(self.W3.data, gain=gain)
        
        
    
    def forward(self,x_in):
        '''note that x0 is useless since it will be removed by boundary identity property'''
        x0,x1,x2,x3 = x_in
        
        '''order 0'''
        I0x = torch.unsqueeze(self.I0@x0,2)
        x0p = torch.inverse(self.D10)@self.B1@x1
        I0xp = torch.unsqueeze(self.I0@x0p,2) # torch.inverse(self.D1)@self.B1 is the projection matrix

        if self.k00 > 0 and self.k0p > 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,X00,I0xp,X0p),2)
        elif  self.k00 > 0 and self.k0p == 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0 = torch.cat((I0x, X00, I0xp ), 2)
        elif self.k00 == 0 and self.k0p > 0:
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,I0xp,X0p),2)
        else: 
            X0 = torch.cat((I0x,I0xp),2)
            
        '''order 1'''
        x1n = self.D20@self.B1.T @torch.inverse(self.D10)@ x0
        I1xn = torch.unsqueeze(self.I1@x1n,2)
        I1x = torch.unsqueeze(self.I1@x1,2)
        x1p = torch.pinverse(self.D1)@self.B2@x2
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
        x2n = self.D2@self.B2.T @torch.pinverse(self.D1)@ x1
        I2xn = torch.unsqueeze(self.I2@x2n,2)
        I2x = torch.unsqueeze(self.I2@x2,2)
        x2p = self.B3@self.D3@x3
        I2xp = torch.unsqueeze(self.I2@x2p,2)
        
        if self.k2n > 0:
            X2nl = chebyshev(self.L2l, self.k2n, x2n)
            X2n = torch.cat((I2xn,X2nl),2)
        else: 
            X2n = I2xn
        
        if self.k21>0 and self.k22>0:
            X2l = chebyshev(self.L2l, self.k21, x2)
            X2u = chebyshev(self.L2u, self.k22, x2)
            X22 = torch.cat((I2x, X2l, X2u),2)
        elif self.k21>0 and self.k22==0:
            X2l = chebyshev(self.L2l, self.k21, x2)
            X22 = torch.cat((I2x, X2l),2)  
        elif self.k21==0 and self.k22>0:
            X2u = chebyshev(self.L2u, self.k22, x2)
            X22 = torch.cat((I2x, X2u),2)
        else:
            X22 = I2x
            
        if self.k2p > 0:
            X2pu = chebyshev(self.L2u, self.k2p, x2p)
            X2p = torch.cat((I2xp, X2pu), 2)
        else:
            X2p = I2xp
            
        X2 = torch.cat((X2n,X22,X2p),2)
            
        '''order 3'''
        x3n = self.B3.T@torch.pinverse(self.D5)@x2
        I3xn = torch.unsqueeze(self.I3@x3n,2)
        I3x = torch.unsqueeze(self.I3@x3,2)
        if self.k3n>0 and self.k33 > 0:
            X3n = chebyshev(self.L3, self.k3n, x3n)
            X33 = chebyshev(self.L3, self.k33, x3)
            X3 = torch.cat((I3xn, X3n, I3x, X33), 2)
        elif self.k3n>0 and self.k33 == 0:
            X3n = chebyshev(self.L3, self.k3n, x3n)
            X3 = torch.cat((I3xn, X3n, I3x), 2)
        elif self.k3n==0 and self.k33 > 0:
            X33 = chebyshev(self.L3, self.k33, x2)
            X3 = torch.cat((I3xn, I3x, X33), 2)   
        else:
            X3 = torch.cat((I3xn,I3x),2)

        y0 = torch.einsum('nik,iok->no',X0,self.W0)
        y1 = torch.einsum('nik,iok->no',X1,self.W1)
        y2 = torch.einsum('nik,iok->no',X2,self.W2)
        y3 = torch.einsum('nik,iok->no',X3,self.W3)

        return y0,y1,y2,y3
    
    
class sccnn_conv_ebli(nn.Module):
    def __init__(self, F_in, F_out, b1, b2, b3, l0, l1l, l1u, l2l, l2u, l3, d1,d2,d3,d5, d10, d20, d30, k00, k0p,k1n,k11,k12,k1p, k2n,k21,k22,k2p,k3n,k33, sigma):
        """
        F_in: number of input features per layer 
        F_out: number of output features per layer
        p: stands for positive, denoting the upper simplex oreder
        n: stands for negative, denoting the lower simplex order
        
        """
        super(sccnn_conv_ebli, self).__init__()
        self.F_in = F_in
        self.F_out = F_out 
        self.B1 = b1 
        self.B2 = b2 
        self.B3 = b3
        self.L0 = l0 
        self.L1l = l1l
        self.L1u = l1u
        self.L1 = self.L1l + self.L1u
        self.L2l = l2l
        self.L2u = l2u   
        self.L2 = self.L2l + self.L2u      
        self.L3 = l3
        self.D1 = d1
        self.D2 = d2
        self.D3 = d3
        self.D5 = d5
        self.D10 = d10
        self.D20 = d20
        self.D30 = d30
        self.sigma = sigma

        self.k00 = k00
        self.k0p = k0p
        
        self.k1n = k1n 
        self.k11 = k11
        self.k12 = k12
        self.k1p = k1p
        
        self.k2n = k2n
        self.k21 = k21
        self.k22 = k22
        self.k2p = k2p
        
        self.k3n = k3n
        self.k33 = k33
        
        self.W0 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k00 + 1+self.k0p)))
        self.W1 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k1n + 1+self.k11+self.k12 + 1+self.k1p)))
        self.W2 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k2n + 1+self.k21+self.k22 + 1+self.k2p)))
        self.W3 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 1+self.k3n + 1+self.k33)))
        
        dim_0 = self.L0.size(dim=0)
        self.I0 = torch.eye(dim_0,device=self.L0.device)
        dim_1 = self.L1l.size(dim=0)
        self.I1 = torch.eye(dim_1,device=self.L1l.device)
        dim_2 = self.L2l.size(dim=0)
        self.I2 = torch.eye(dim_2,device=self.L2l.device)
        dim_3 = self.L3.size(dim=0)
        self.I3 = torch.eye(dim_3,device=self.L3.device)
        
        self.reset_parameters()
        print("created SCCNN layers")
    
    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W0.data, gain=gain)
        nn.init.xavier_uniform_(self.W1.data, gain=gain)
        nn.init.xavier_uniform_(self.W2.data, gain=gain)
        nn.init.xavier_uniform_(self.W3.data, gain=gain)
        
        
    
    def forward(self,x_in):
        '''note that x0 is useless since it will be removed by boundary identity property'''
        x0,x1,x2,x3 = x_in
        
        '''order 0'''
        I0x = torch.unsqueeze(self.I0@x0,2)
        x0p = torch.inverse(self.D10)@self.B1@x1
        I0xp = torch.unsqueeze(self.I0@x0p,2) # torch.inverse(self.D1)@self.B1 is the projection matrix

        if self.k00 > 0 and self.k0p > 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,X00,I0xp,X0p),2)
        elif  self.k00 > 0 and self.k0p == 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0 = torch.cat((I0x, X00, I0xp ), 2)
        elif self.k00 == 0 and self.k0p > 0:
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,I0xp,X0p),2)
        else: 
            X0 = torch.cat((I0x,I0xp),2)
            
        '''order 1'''
        x1n = self.D20@self.B1.T @torch.inverse(self.D10)@ x0
        I1xn = torch.unsqueeze(self.I1@x1n,2)
        I1x = torch.unsqueeze(self.I1@x1,2)
        x1p = torch.pinverse(self.D1)@self.B2@x2
        I1xp = torch.unsqueeze(self.I1@x1p,2)
        
        if self.k1n > 0:
            X1nl = chebyshev(self.L1l, self.k1n, x1n)
            X1n = torch.cat((I1xn,X1nl),2)
        else: 
            X1n = I1xn
        
        if self.k11>0 and self.k12>0:
            X1l = chebyshev(self.L1, self.k11+self.k12, x1)
            # X1u = chebyshev(self.L1u, self.k12, x1)
            X11 = torch.cat((I1x, X1l),2)
        # elif self.k11>0 and self.k12==0:
        #     X1l = chebyshev(self.L1l, self.k11, x1)
        #     X11 = torch.cat((I1x, X1l),2)  
        # elif self.k11==0 and self.k12>0:
        #     X1u = chebyshev(self.L1u, self.k12, x1)
        #     X11 = torch.cat((I1x, X1u),2)
        else:
            X11 = I1x
            
        if self.k1p > 0:
            X1pu = chebyshev(self.L1u, self.k1p, x1p)
            X1p = torch.cat((I1xp, X1pu), 2)
        else:
            X1p = I1xp
            
        X1 = torch.cat((X1n,X11,X1p),2)
        
        '''order 2'''
        x2n = self.D2@self.B2.T @torch.pinverse(self.D1)@ x1
        I2xn = torch.unsqueeze(self.I2@x2n,2)
        I2x = torch.unsqueeze(self.I2@x2,2)
        x2p = self.B3@self.D3@x3
        I2xp = torch.unsqueeze(self.I2@x2p,2)
        
        if self.k2n > 0:
            X2nl = chebyshev(self.L2l, self.k2n, x2n)
            X2n = torch.cat((I2xn,X2nl),2)
        else: 
            X2n = I2xn
        
        if self.k21>0 and self.k22>0:
            X2l = chebyshev(self.L2, self.k21+self.k22, x2)
            # X2u = chebyshev(self.L2u, self.k22, x2)
            X22 = torch.cat((I2x, X2l),2)
        # elif self.k21>0 and self.k22==0:
        #     X2l = chebyshev(self.L2l, self.k21, x2)
        #     X22 = torch.cat((I2x, X2l),2)  
        # elif self.k21==0 and self.k22>0:
        #     X2u = chebyshev(self.L2u, self.k22, x2)
        #     X22 = torch.cat((I2x, X2u),2)
        else:
            X22 = I2x
            
        if self.k2p > 0:
            X2pu = chebyshev(self.L2u, self.k2p, x2p)
            X2p = torch.cat((I2xp, X2pu), 2)
        else:
            X2p = I2xp
            
        X2 = torch.cat((X2n,X22,X2p),2)
            
        '''order 3'''
        x3n = self.B3.T@torch.pinverse(self.D5)@x2
        I3xn = torch.unsqueeze(self.I3@x3n,2)
        I3x = torch.unsqueeze(self.I3@x3,2)
        if self.k3n>0 and self.k33 > 0:
            X3n = chebyshev(self.L3, self.k3n, x3n)
            X33 = chebyshev(self.L3, self.k33, x3)
            X3 = torch.cat((I3xn, X3n, I3x, X33), 2)
        elif self.k3n>0 and self.k33 == 0:
            X3n = chebyshev(self.L3, self.k3n, x3n)
            X3 = torch.cat((I3xn, X3n, I3x), 2)
        elif self.k3n==0 and self.k33 > 0:
            X33 = chebyshev(self.L3, self.k33, x2)
            X3 = torch.cat((I3xn, I3x, X33), 2)   
        else:
            X3 = torch.cat((I3xn,I3x),2)

        y0 = torch.einsum('nik,iok->no',X0,self.W0)
        y1 = torch.einsum('nik,iok->no',X1,self.W1)
        y2 = torch.einsum('nik,iok->no',X2,self.W2)
        y3 = torch.einsum('nik,iok->no',X3,self.W3)
        y0 = self.sigma(y0)
        y1 = self.sigma(y1)
        y2 = self.sigma(y2)
        y3 = self.sigma(y3)
        return y0,y1,y2,y3
    

class sccnn_conv_no_b3(nn.Module):
    def __init__(self, F_in, F_out, b1, b2, b3, l0, l1l, l1u, l2l, l2u, l3, d1,d2,d3,d5, d10, d20, d30, k00, k0p,k1n,k11,k12,k1p, k2n,k21,k22,k2p,k3n,k33, sigma):
        """
        F_in: number of input features per layer 
        F_out: number of output features per layer
        p: stands for positive, denoting the upper simplex oreder
        n: stands for negative, denoting the lower simplex order
        
        """
        super(sccnn_conv_no_b3, self).__init__()
        self.F_in = F_in
        self.F_out = F_out 
        self.B1 = b1 
        self.B2 = b2 
        self.B3 = b3
        self.L0 = l0 
        self.L1l = l1l
        self.L1u = l1u
        self.L1 = self.L1l + self.L1u
        self.L2l = l2l
        self.L2u = l2u   
        self.L2 = self.L2l + self.L2u      
        self.L3 = l3
        self.D1 = d1
        self.D2 = d2
        self.D3 = d3
        self.D5 = d5
        self.D10 = d10
        self.D20 = d20
        self.D30 = d30
        self.sigma = sigma

        self.k00 = k00
        self.k0p = k0p
        
        self.k1n = k1n 
        self.k11 = k11
        self.k12 = k12
        self.k1p = k1p
        
        self.k2n = k2n
        self.k21 = k21
        self.k22 = k22
        self.k2p = k2p
        
        self.k3n = k3n
        self.k33 = k33
        
        self.W0 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k00 + 1+self.k0p)))
        self.W1 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k1n + 1+self.k11+self.k12 + 1+self.k1p)))
        self.W2 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k2n + 1+self.k21 )))
        # self.W3 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k3n + 1+self.k33)))
        
        dim_0 = self.L0.size(dim=0)
        self.I0 = torch.eye(dim_0,device=self.L0.device)
        dim_1 = self.L1l.size(dim=0)
        self.I1 = torch.eye(dim_1,device=self.L1l.device)
        dim_2 = self.L2l.size(dim=0)
        self.I2 = torch.eye(dim_2,device=self.L2l.device)
        dim_3 = self.L3.size(dim=0)
        self.I3 = torch.eye(dim_3,device=self.L3.device)
        
        self.reset_parameters()
        print("created SCCNN layers")
    
    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W0.data, gain=gain)
        nn.init.xavier_uniform_(self.W1.data, gain=gain)
        nn.init.xavier_uniform_(self.W2.data, gain=gain)
        # nn.init.xavier_uniform_(self.W3.data, gain=gain)
        
        
    
    def forward(self,x_in):
        '''note that x0 is useless since it will be removed by boundary identity property'''
        x0,x1,x2 = x_in
        
        '''order 0'''
        I0x = torch.unsqueeze(self.I0@x0,2)
        x0p = torch.inverse(self.D10)@self.B1@x1
        I0xp = torch.unsqueeze(self.I0@x0p,2) # torch.inverse(self.D1)@self.B1 is the projection matrix

        if self.k00 > 0 and self.k0p > 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,X00,I0xp,X0p),2)
        elif  self.k00 > 0 and self.k0p == 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0 = torch.cat((I0x, X00, I0xp ), 2)
        elif self.k00 == 0 and self.k0p > 0:
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,I0xp,X0p),2)
        else: 
            X0 = torch.cat((I0x,I0xp),2)
            
        '''order 1'''
        x1n = self.D20@self.B1.T @torch.inverse(self.D10)@ x0
        I1xn = torch.unsqueeze(self.I1@x1n,2)
        I1x = torch.unsqueeze(self.I1@x1,2)
        x1p = torch.pinverse(self.D1)@self.B2@x2
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
        x2n = self.D2@self.B2.T @torch.pinverse(self.D1)@ x1
        I2xn = torch.unsqueeze(self.I2@x2n,2)
        I2x = torch.unsqueeze(self.I2@x2,2)
        # x2p = self.B3@self.D3@x3
        # I2xp = torch.unsqueeze(self.I2@x2p,2)
        
        if self.k2n > 0:
            X2nl = chebyshev(self.L2l, self.k2n, x2n)
            X2n = torch.cat((I2xn,X2nl),2)
        else: 
            X2n = I2xn
        
        if self.k21>0 and self.k22>0:
            X2l = chebyshev(self.L2l, self.k21, x2)
            # X2u = chebyshev(self.L2u, self.k22, x2)
            X22 = torch.cat((I2x, X2l),2)
        elif self.k21>0 and self.k22==0:
            X2l = chebyshev(self.L2l, self.k21, x2)
            X22 = torch.cat((I2x, X2l),2)  
        # elif self.k21==0 and self.k22>0:
        #     X2u = chebyshev(self.L2u, self.k22, x2)
        #     X22 = torch.cat((I2x, X2u),2)
        else:
            X22 = I2x
            
        # if self.k2p > 0:
        #     X2pu = chebyshev(self.L2u, self.k2p, x2p)
        #     X2p = torch.cat((I2xp, X2pu), 2)
        # else:
        #     X2p = I2xp
            
        X2 = torch.cat((X2n,X22),2)
            
        # '''order 3'''
        # x3n = self.B3.T@torch.pinverse(self.D5)@x2
        # I3xn = torch.unsqueeze(self.I3@x3n,2)
        # I3x = torch.unsqueeze(self.I3@x3,2)
        # if self.k3n>0 and self.k33 > 0:
        #     X3n = chebyshev(self.L3, self.k3n, x3n)
        #     X33 = chebyshev(self.L3, self.k33, x3)
        #     X3 = torch.cat((I3xn, X3n, I3x, X33), 2)
        # elif self.k3n>0 and self.k33 == 0:
        #     X3n = chebyshev(self.L3, self.k3n, x3n)
        #     X3 = torch.cat((I3xn, X3n, I3x), 2)
        # elif self.k3n==0 and self.k33 > 0:
        #     X33 = chebyshev(self.L3, self.k33, x2)
        #     X3 = torch.cat((I3xn, I3x, X33), 2)   
        # else:
        #     X3 = torch.cat((I3xn,I3x),2)

        y0 = torch.einsum('nik,iok->no',X0,self.W0)
        y1 = torch.einsum('nik,iok->no',X1,self.W1)
        y2 = torch.einsum('nik,iok->no',X2,self.W2)
        # y3 = torch.einsum('nik,iok->no',X3,self.W3)
        y0 = self.sigma(y0)
        y1 = self.sigma(y1)
        y2 = self.sigma(y2)
        # y3 = self.sigma(y3)
        return y0,y1,y2
    
    
class sccnn_conv_no_b2(nn.Module):
    def __init__(self, F_in, F_out, b1, b2, b3, l0, l1l, l1u, l2l, l2u, l3, d1,d2,d3,d5, d10, d20, d30, k00, k0p,k1n,k11,k12,k1p, k2n,k21,k22,k2p,k3n,k33, sigma):
        """
        F_in: number of input features per layer 
        F_out: number of output features per layer
        p: stands for positive, denoting the upper simplex oreder
        n: stands for negative, denoting the lower simplex order
        
        """
        super(sccnn_conv_no_b2, self).__init__()
        self.F_in = F_in
        self.F_out = F_out 
        self.B1 = b1 
        self.B2 = b2 
        self.B3 = b3
        self.L0 = l0 
        self.L1l = l1l
        self.L1u = l1u
        self.L1 = self.L1l + self.L1u
        self.L2l = l2l
        self.L2u = l2u   
        self.L2 = self.L2l + self.L2u      
        self.L3 = l3
        self.D1 = d1
        self.D2 = d2
        self.D3 = d3
        self.D5 = d5
        self.D10 = d10
        self.D20 = d20
        self.D30 = d30
        self.sigma = sigma

        self.k00 = k00
        self.k0p = k0p
        
        self.k1n = k1n 
        self.k11 = k11
        self.k12 = k12
        self.k1p = k1p
        
        self.k2n = k2n
        self.k21 = k21
        self.k22 = k22
        self.k2p = k2p
        
        self.k3n = k3n
        self.k33 = k33
        
        self.W0 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k00 + 1+self.k0p)))
        self.W1 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k1n + 1+self.k11)))
        #self.W2 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k2n + 1+self.k21 )))
        # self.W3 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out,1+self.k3n + 1+self.k33)))
        
        dim_0 = self.L0.size(dim=0)
        self.I0 = torch.eye(dim_0,device=self.L0.device)
        dim_1 = self.L1l.size(dim=0)
        self.I1 = torch.eye(dim_1,device=self.L1l.device)
        dim_2 = self.L2l.size(dim=0)
        self.I2 = torch.eye(dim_2,device=self.L2l.device)
        dim_3 = self.L3.size(dim=0)
        self.I3 = torch.eye(dim_3,device=self.L3.device)
        
        self.reset_parameters()
        print("created SCCNN layers")
    
    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W0.data, gain=gain)
        nn.init.xavier_uniform_(self.W1.data, gain=gain)
        # nn.init.xavier_uniform_(self.W2.data, gain=gain)
        # nn.init.xavier_uniform_(self.W3.data, gain=gain)
        
        
    
    def forward(self,x_in):
        '''note that x0 is useless since it will be removed by boundary identity property'''
        x0,x1 = x_in
        
        '''order 0'''
        I0x = torch.unsqueeze(self.I0@x0,2)
        x0p = torch.inverse(self.D10)@self.B1@x1
        I0xp = torch.unsqueeze(self.I0@x0p,2) # torch.inverse(self.D1)@self.B1 is the projection matrix

        if self.k00 > 0 and self.k0p > 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,X00,I0xp,X0p),2)
        elif  self.k00 > 0 and self.k0p == 0:
            X00 = chebyshev(self.L0, self.k00, x0)
            X0 = torch.cat((I0x, X00, I0xp ), 2)
        elif self.k00 == 0 and self.k0p > 0:
            X0p = chebyshev(self.L0, self.k0p, x0p) 
            X0 = torch.cat((I0x,I0xp,X0p),2)
        else: 
            X0 = torch.cat((I0x,I0xp),2)
            
        '''order 1'''
        x1n = self.D20@self.B1.T @torch.inverse(self.D10)@ x0
        I1xn = torch.unsqueeze(self.I1@x1n,2)
        I1x = torch.unsqueeze(self.I1@x1,2)
        # x1p = torch.pinverse(self.D1)@self.B2@x2
        # I1xp = torch.unsqueeze(self.I1@x1p,2)
        
        if self.k1n > 0:
            X1nl = chebyshev(self.L1l, self.k1n, x1n)
            X1n = torch.cat((I1xn,X1nl),2)
        else: 
            X1n = I1xn
        
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
            
        X1 = torch.cat((X1n,X11),2)
        
        # '''order 2'''
        # x2n = self.D2@self.B2.T @torch.pinverse(self.D1)@ x1
        # I2xn = torch.unsqueeze(self.I2@x2n,2)
        # I2x = torch.unsqueeze(self.I2@x2,2)
        # # x2p = self.B3@self.D3@x3
        # # I2xp = torch.unsqueeze(self.I2@x2p,2)
        
        # if self.k2n > 0:
        #     X2nl = chebyshev(self.L2l, self.k2n, x2n)
        #     X2n = torch.cat((I2xn,X2nl),2)
        # else: 
        #     X2n = I2xn
        
        # if self.k21>0 and self.k22>0:
        #     X2l = chebyshev(self.L2l, self.k21, x2)
        #     # X2u = chebyshev(self.L2u, self.k22, x2)
        #     X22 = torch.cat((I2x, X2l),2)
        # elif self.k21>0 and self.k22==0:
        #     X2l = chebyshev(self.L2l, self.k21, x2)
        #     X22 = torch.cat((I2x, X2l),2)  
        # # elif self.k21==0 and self.k22>0:
        # #     X2u = chebyshev(self.L2u, self.k22, x2)
        # #     X22 = torch.cat((I2x, X2u),2)
        # else:
        #     X22 = I2x
            
        # # if self.k2p > 0:
        # #     X2pu = chebyshev(self.L2u, self.k2p, x2p)
        # #     X2p = torch.cat((I2xp, X2pu), 2)
        # # else:
        # #     X2p = I2xp
            
        # X2 = torch.cat((X2n,X22),2)
            
        # '''order 3'''
        # x3n = self.B3.T@torch.pinverse(self.D5)@x2
        # I3xn = torch.unsqueeze(self.I3@x3n,2)
        # I3x = torch.unsqueeze(self.I3@x3,2)
        # if self.k3n>0 and self.k33 > 0:
        #     X3n = chebyshev(self.L3, self.k3n, x3n)
        #     X33 = chebyshev(self.L3, self.k33, x3)
        #     X3 = torch.cat((I3xn, X3n, I3x, X33), 2)
        # elif self.k3n>0 and self.k33 == 0:
        #     X3n = chebyshev(self.L3, self.k3n, x3n)
        #     X3 = torch.cat((I3xn, X3n, I3x), 2)
        # elif self.k3n==0 and self.k33 > 0:
        #     X33 = chebyshev(self.L3, self.k33, x2)
        #     X3 = torch.cat((I3xn, I3x, X33), 2)   
        # else:
        #     X3 = torch.cat((I3xn,I3x),2)

        y0 = torch.einsum('nik,iok->no',X0,self.W0)
        y1 = torch.einsum('nik,iok->no',X1,self.W1)
        # y2 = torch.einsum('nik,iok->no',X2,self.W2)
        # y3 = torch.einsum('nik,iok->no',X3,self.W3)
        y0 = self.sigma(y0)
        y1 = self.sigma(y1)
        # y2 = self.sigma(y2)
        # y3 = self.sigma(y3)
        return y0,y1
    