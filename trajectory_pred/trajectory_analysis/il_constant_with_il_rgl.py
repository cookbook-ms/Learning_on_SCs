'''
This script is to study the integral lipschitz constants of the filters in the SCNN learned with the regularizer
'''


import os, sys
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import matplotlib as mpl

from synthetic_data_gen import load_dataset

num_feat = 16

model_name = 'scnn31'
model_path = './models/std_normalized_il_study_'+model_name+'_il_'+ model_name+'_500.npy'
weights = np.load(model_path, allow_pickle=True)
# print(weights.shape)
# print([weights[i].shape for i in range(weights.shape[0])])
# print([weights[i][0][0] for i in range(5)])

order = 2
 # Load data
folder = 'trajectory_data_1hop_working'
X, B_matrices, y, train_mask, test_mask, G_undir, last_nodes, target_nodes = load_dataset(folder)
B1, B2 = B_matrices
# compute the diagonal matrices in the hodge laplacian normalization 
D2 = np.maximum(np.diag(abs(B2)@np.ones(B2.shape[1])),np.identity(B2.shape[0]))
D1 = 2 * np.diag(abs(B1)@D2@np.ones(D2.shape[0]))
D3 = 1/3*np.identity(B2.shape[1])
L1_lower = D2 @ B1.T @ la.pinv(D1) @ B1 
L1_upper = B2 @ D3 @ B2.T @ la.inv(D2)
lam_l, v  = la.eig(L1_lower)
lam_u, v = la.eig(L1_upper)
num_edges = L1_lower.shape[0]
print(np.max(lam_l.real), np.max(lam_u.real))

layer_ids = [0,1,2]
for layer_id in layer_ids:
    if layer_id == 0:
        fin = 1 
        c_layer1 = np.empty((1,2,fin*num_feat))
    elif layer_id == 1:
        fin = 16
        c_layer2 = np.empty((1,2,fin*num_feat))
    elif layer_id == 2:
        fin = 16
        c_layer3 = np.empty((1,2,fin*num_feat))

    il_constant_l = np.empty((num_edges,fin,num_feat))
    il_constant_u = np.empty((num_edges,fin,num_feat))
    for feature_id_in in range(fin):
        for feature_id_out1 in range(0,num_feat,1):
            coeff_l = [weights[(layer_id-1)*(2*order+1)+i][feature_id_in][feature_id_out1] for i in range(order+1)]
            a = list(np.linspace(order+1,2*order,order))
            a.insert(0,0)
            # print(a)
            coeff_u = [weights[(layer_id-1)*(2*order+1)+int(i)][feature_id_in][feature_id_out1] for i in a]        
            lam_k_l = np.array([lam_l**k for k in range(order+1)])
            lam_k_u = np.array([lam_u**k for k in range(order+1)])
            # print(lam_k)
            func_l = lam_k_l.T@coeff_l 
            func_u = lam_k_u.T@coeff_u
            il_constant_l[:,feature_id_in,feature_id_out1] = func_l
            il_constant_u[:,feature_id_in,feature_id_out1] = func_u
            c1 = np.amax(abs(il_constant_l),0)
            c2 = np.amax(abs(il_constant_u),0)
            
            if layer_id == 0:
                c_layer1[0,0,:] = np.ravel(c1)
                c_layer1[0,1,:] = np.ravel(c2)
            if layer_id == 1:
                c_layer2[0,0,:] = np.ravel(c1)
                c_layer2[0,1,:] = np.ravel(c2)
            if layer_id == 2:
                c_layer3[0,0,:] = np.ravel(c1)
                c_layer3[0,1,:] = np.ravel(c2)

mpl.rc('text', usetex = True)
mpl.rc('font', **{'family' : "sans-serif"})
params= {'text.latex.preamble' : r'\usepackage{amsmath}'}
plt.rcParams.update(params)
plt.rcParams['figure.figsize'] = 4.5,3
fs = 20
plt.rcParams['xtick.labelsize'] = fs 
plt.rcParams['ytick.labelsize'] = fs 

fig,ax = plt.subplots(sharex=True)
ax.boxplot([c_layer1[0,0,:], c_layer1[0,1,:], c_layer2[0,0,:], c_layer2[0,1,:],c_layer3[0,0,:], c_layer3[0,1,:],]) 
 
plt.xticks([1, 2, 3, 4, 5, 6], ['$C^1_'r'\text{1,d}$','$C^1_'r'\text{1,u}$','$C^2_'r'\text{1,d}$','$C^2_'r'\text{1,u}$','$C^3_'r'\text{1,d}$','$C^3_'r'\text{1,u}$'])  
      
# ax.set_ylim(bottom=0)

plt.savefig('./il_constant_plot/il_'+model_name+'.pdf', format="pdf", bbox_inches="tight")