#!/usr/bin/env python3.8.5
'''study pert influence on dist'''
from re import A
import torch
import torch.nn as nn
import torch.nn.functional
import torch.utils.data as data
import numpy.linalg as la
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import matplotlib.pyplot as plt
import itertools
import sys
sys.path.append('.')
import argparse
import os
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rc('font', **{'family' : "sans-serif"})
params= {'text.latex.preamble' : r'\usepackage{amsmath}'}
plt.rcParams.update(params)
fs = 20
plt.rcParams['xtick.labelsize'] = fs 
plt.rcParams['ytick.labelsize'] = fs 

'''nn'''
fs = 20

num_perts = 10
pert_list = np.linspace(0,1,num_perts) 

for num_layers in [0,1,2,3,4]:
    print(num_layers)
    for hidden_features in [16,32]:
        for model_name in ['sccnn_node_pert_0_no_b2','sccnn_node_pert_0']: #'sccnn_node_pert_1_no_b2','sccnn_node_pert_1','sccnn_node_pert_2'
            var_path = r'./pert_var_nn_' + model_name      
            savepath = r'./pert_plots/'+ model_name +'/layer'+str(num_layers)  + '_' + str(hidden_features)
            if not os.path.exists(savepath):
                os.makedirs(savepath)
                
            for K in [0,1,2,3,4,5]:
                fig, ax = plt.subplots()
                var_save_path = var_path+'/%dlayers_%dorders_%dfeatures.npy'%(num_layers,K,hidden_features)  
                with open(var_save_path, 'rb') as f:
                    auc = np.load(f)
                    auc_pert = np.load(f)
                    dist = np.load(f)   
                    dist_edge = np.load(f)
                    if model_name in ['sccnn_node_pert_0']:
                        dist_tri = np.load(f)   
                        epsilon_0 = np.load(f)   
                        epsilon_rlt_0 = np.load(f)   
                        epsilon_rlt_1l = np.load(f)   
                        epsilon_10 = np.load(f)    
                        epsilon_01 = np.load(f)   
                        # print('xxxxx',dist,dist_edge)
                    if model_name in ['sccnn_node_pert_0_no_b2']:
                        epsilon_0 = np.load(f)   
                        epsilon_rlt_0 = np.load(f)   
                        epsilon_rlt_1l = np.load(f)   
                        epsilon_10 = np.load(f)    
                        epsilon_01 = np.load(f)  
                    if model_name in ['sccnn_node_pert_1']:
                        dist_tri = np.load(f)   
                        epsilon_1 = np.load(f)   
                        epsilon_rlt_0 = np.load(f)   
                        epsilon_rlt_1l = np.load(f)   
                        epsilon_rlt_1u = np.load(f)   
                        epsilon_rlt_2 = np.load(f)   
                        epsilon_01 = np.load(f)    
                        epsilon_12 = np.load(f)  
                    if model_name in ['sccnn_node_pert_1_no_b2']:  
                        epsilon_1 = np.load(f)   
                        epsilon_rlt_0 = np.load(f)   
                        epsilon_rlt_1l = np.load(f)    
                        epsilon_01 = np.load(f) 
                    if model_name in ['sccnn_node_pert_2']:
                        dist_tri = np.load(f)   
                        epsilon_2 = np.load(f)   
                        epsilon_rlt_1u = np.load(f)   
                        epsilon_rlt_2 = np.load(f)   
                        epsilon_12 = np.load(f)    
                        epsilon_21 = np.load(f)        
            
                dist_node_line = ax.plot(pert_list,dist.mean(1),linewidth=3,label='dist node')
                dist_edge_line = ax.plot(pert_list,dist_edge.mean(1),linewidth=3,label='dist edge')
                if model_name in ['sccnn_node_pert_0','sccnn_node_pert_1','sccnn_node_pert_2']:
                    dist_tri_line = ax.plot(pert_list,dist_tri.mean(1),linewidth=3,label='dist tri') 
                # if model_name in ['sccnn_node_pert_0_no_b2','sccnn_node_pert_0']:
                #     ax.set_xlabel('$\epsilon_'r'\text{0}$',fontsize=fs)
                # elif model_name in ['sccnn_node_pert_1_no_b2','sccnn_node_pert_1']:
                #     ax.set_xlabel('$\epsilon_'r'\text{1}$',fontsize=fs)
                # else:
                ax.set_xlabel('$\epsilon_'r'\text{0}$',fontsize=fs)
                ax.legend(fontsize=fs)
                ax.set_yscale('symlog',linthresh=1e-3)
                ax.set_xscale('symlog',linthresh=1e-3)
                # # #ax.set_xlim(np.arange(0,10,step=1),fontsize=fs)
                # # #ax.set_title('sccnn%d'%(K))
                # # # fig.colorbar(surf)
                # plt.show()
                plt.savefig(savepath+'/%dlayers_%dorders_%dfeatures.pdf'%(num_layers,K,hidden_features), format="pdf", bbox_inches="tight")       