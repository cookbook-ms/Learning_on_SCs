#!/usr/bin/env python3.8.5

import torch
import torch.nn as nn
import torch.nn.functional
import torch.linalg as tla
import torch.utils.data as data
import numpy.linalg as la
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import itertools
import sys
sys.path.append('.')
import argparse
import os

from tri_predictor import compute_auc, compute_loss, MLPPredictor
'''for filters'''
from sccnn_model import sccnn
from sccnn_stability import sccnn_stability

def main():
    '''
    hyperparameter
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-hd", "--hidden_features", help="the number of hidden features", type=int, default=32)
    parser.add_argument("-m","--model_name",help="the model name", type=str, default='sccnn_node')
    parser.add_argument("-l", "--layers", help="the number of layers", type=int, default=1) # note that here it is the number of intermediate layers
    parser.add_argument("-k","--filter_order_same",help="the filter order of the model all filters", type=int, default=2)
    parser.add_argument("-k00","--filter_order_H00",help="the filter order of the model H00", type=int, default=1)
    parser.add_argument("-k0p","--filter_order_H0p",help="the filter order of the model H0n", type=int, default=1)
    parser.add_argument("-k1n","--filter_order_H1n",help="the filter order of H1p", type=int, default=1)
    parser.add_argument("-k11","--filter_order_H11_lower",help="the lower filter order of H11", type=int, default=1)
    parser.add_argument("-k12","--filter_order_H11_upper",help="the upper filter order of H11", type=int, default=1)
    parser.add_argument("-k1p","--filter_order_H1p",help="the filter order of H1p", type=int, default=1)
    parser.add_argument("-k2n","--filter_order_H2n",help="the filter order of the model H2n", type=int, default=1)
    parser.add_argument("-k22","--filter_order_H22",help="the filter order of the model H22", type=int, default=1)
       
    parser.add_argument('-lr',"--learning_rate",help="learning rate of adam", default=0.001)
    parser.add_argument('-e',"--epochs",help="number of epochs", type=int, default=1000)
    parser.add_argument("-a", "--activations", help="activation function", type=str, default='leaky_relu')
    parser.add_argument("-rlz", "--realizations", help="number of realizations", type=int, default=1)
    args = parser.parse_args()

    activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'elu': nn.ELU(),
        'selu': nn.SELU(),
        'leaky_relu' : nn.LeakyReLU(0.01),
    }
    
    prefix = './data/s2_3_collaboration_complex'
    starting_node = '150250' 

    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    print(device)
    topdim = 5
    boundaries = np.load('{}/{}_boundaries.npy'.format(prefix,starting_node),allow_pickle=True)

    boundaries = [boundaries[i].toarray() for i in range(topdim+1)]
    test_perc = 0.10
    val_perc = 0.10
    # edge-to-triangle incidence matrix 
    b2 = boundaries[1]
    b1 = boundaries[0] 
    print(b2.shape)
    # triangle ids
    tri_idx = np.arange(b2.shape[1])
    edge_idx = np.reshape(np.nonzero(b2.T)[1],(-1,3)) 
    node_idx = np.reshape(np.nonzero(b1.T)[1],(-1,2))

    # the underlying true cochains
    cochains_dic = np.load('{}/{}_cochains.npy'.format(prefix,starting_node),allow_pickle=True)
    cochains =[list(cochains_dic[i].values()) for i in range(len(cochains_dic))]
    f_tri = np.array(cochains[2])
    a = f_tri<=7

    num_perts = 10
    pert_list_2 = np.linspace(0,1,num_perts) # perturbation on L1l   
    #pert_list_2 = np.logspace(-3,0,num_perts) 
    num_rlz = 10
    rlz_idx = np.arange(1,1+num_rlz)
    
    dist = np.empty((len(pert_list_2),len(rlz_idx)))
    dist_edge = np.empty((len(pert_list_2),len(rlz_idx)))
    dist_tri = np.empty((len(pert_list_2),len(rlz_idx)))
    auc = np.empty((len(pert_list_2),len(rlz_idx)))
    auc_pert = np.empty((len(pert_list_2),len(rlz_idx)))
    epsilon_2 = np.empty((len(pert_list_2),len(rlz_idx)))
    epsilon_rlt_1u = np.empty((len(pert_list_2),len(rlz_idx)))
    epsilon_rlt_2 = np.empty((len(pert_list_2),len(rlz_idx))) 
    epsilon_21 = np.empty((len(pert_list_2),len(rlz_idx)))
    epsilon_12 = np.empty((len(pert_list_2),len(rlz_idx)))

    for j in np.arange(len(rlz_idx)):
        args.realizations = rlz_idx[j]
        torch.manual_seed(1337*args.realizations)
        np.random.seed(1337*args.realizations)
        neg_idx = np.random.permutation(np.where(a!=0)[0])
        pos_idx = np.random.permutation(np.where(a==0)[0])

        pos_size = len(pos_idx)
        neg_size = len(neg_idx)
        pos_i, pos_j, pos_k = edge_idx[pos_idx][:,0], edge_idx[pos_idx][:,1], edge_idx[pos_idx][:,2]
        pos_node_i, pos_node_j,pos_node_k = node_idx[pos_i][:,0],node_idx[pos_j][:,1],node_idx[pos_k][:,0]

        neg_i, neg_j, neg_k = edge_idx[neg_idx][:,0], edge_idx[neg_idx][:,1], edge_idx[neg_idx][:,2]
        neg_node_i, neg_node_j,neg_node_k = node_idx[neg_i][:,0],node_idx[neg_j][:,1],node_idx[neg_k][:,0]

        test_size_pos = int(pos_size * test_perc)
        train_size_pos = pos_size - test_size_pos
        val_size_pos = int(pos_size * val_perc)
        train_size_pos = train_size_pos - val_size_pos
        train_pos_i, train_pos_j, train_pos_k = pos_node_i[:train_size_pos], pos_node_j[:train_size_pos], pos_node_k[:train_size_pos]
        val_pos_i, val_pos_j, val_pos_k = pos_node_i[train_size_pos:(train_size_pos+val_size_pos)], pos_node_j[train_size_pos:(train_size_pos+val_size_pos)], pos_node_k[train_size_pos:(train_size_pos+val_size_pos)] 
        
        test_pos_i, test_pos_j, test_pos_k = pos_node_i[train_size_pos:], pos_node_j[train_size_pos:], pos_node_k[train_size_pos:] 

        test_size_neg = int(neg_size * test_perc)
        train_size_neg = neg_size - test_size_neg
        val_size_neg = int(neg_size * val_perc)
        train_size_neg = train_size_neg - val_size_neg
        
        train_neg_i, train_neg_j, train_neg_k = neg_node_i[:train_size_neg], neg_node_j[:train_size_neg], neg_node_k[:train_size_neg]
        
        val_neg_i, val_neg_j, val_neg_k = neg_node_i[train_size_neg:(train_size_neg+val_size_neg)], neg_node_j[train_size_neg:(train_size_neg+val_size_neg)], neg_node_k[train_size_neg:(train_size_neg+val_size_neg)] 
        
        test_neg_i, test_neg_j, test_neg_k = neg_node_i[train_size_neg:], neg_node_j[train_size_neg:], neg_node_k[train_size_neg:] 

        b2_train = b2[:,pos_idx][:,:train_size_pos]
        # Check if the training incidece matrix contains open triangles 

        '''L1'''
        D2 = np.maximum(np.diag(abs(b2_train)@np.ones(b2_train.shape[1])),np.identity(b2_train.shape[0]))
        D1 = 2 * np.diag(abs(b1)@D2@np.ones(D2.shape[0]))
        D3 = 1/3*np.identity(b2_train.shape[1])
        L1l = sla.fractional_matrix_power(D2,0.5) @ b1.T @ la.pinv(D1) @ b1 @ sla.fractional_matrix_power(D2,0.5)
        L1u = la.inv(sla.fractional_matrix_power(D2,0.5)) @ b2_train @ D3 @ b2_train.T @ la.inv(sla.fractional_matrix_power(D2,0.5))
        L1 = L1l + L1u

        '''L0'''
        b1_train = b1
        D20 = np.maximum(np.diag(abs(b1_train)@np.ones(b1_train.shape[1])),np.identity(b1_train.shape[0]))
        L0 =  la.inv(sla.fractional_matrix_power(D20,0.5)) @ b1_train @ b1_train.T @ la.inv(sla.fractional_matrix_power(D20,0.5)) 
        
        '''L2'''
        D5 = np.diag(abs(b2_train)@np.ones(b2_train.shape[1]))
        L2 = b2_train.T @la.pinv(D5)@ b2_train
        
        L0 = torch.tensor(L0,dtype=torch.float32,device=device)
        B1 = torch.tensor(b1,dtype=torch.float32,device=device)
        B2_train = torch.tensor(b2_train,dtype=torch.float32,device=device)
        L1l = torch.tensor(L1l,dtype=torch.float32,device=device)
        L1u = torch.tensor(L1u,dtype=torch.float32,device=device)
        L1 = torch.tensor(L1,dtype=torch.float32,device=device)
        L2 = torch.tensor(L2,dtype=torch.float32,device=device)
        
        D1 = torch.tensor(D1,dtype=torch.float32,device=device)    
        D2 = torch.tensor(D2,dtype=torch.float32,device=device)    
        D3 = torch.tensor(D3,dtype=torch.float32,device=device)    
        D5 = torch.tensor(D5,dtype=torch.float32,device=device)    

        sigma = activations[args.activations]
        hidden_features = args.hidden_features

        k = args.filter_order_same
        k00 = k
        k0p = k
        k1n = k
        k11 = k
        k12 = k
        k1p = k
        k2n = k
        k22 = k
        model_name = args.model_name
        print(model_name)

        F_intermediate = []
        for i in range(args.layers):
            F_intermediate.append(hidden_features)
    
        # use the edge flow as the input feature to perform triangle prediction
        f_edge = torch.tensor(np.array(cochains[1]),dtype=torch.float32)
        f_edge.resize_(f_edge.shape[0],1)
        f_edge = f_edge.to(device)
        f_node = torch.tensor(np.array(cochains[0]),dtype=torch.float32)
        f_node.resize_(f_node.shape[0],1)
        f_node = f_node.to(device)
        f_tri = torch.tensor(np.zeros(b2_train.shape[1]),dtype=torch.float32)
        f_tri.resize_(f_tri.shape[0],1)
        f_tri = f_tri.to(device)

        f_input = (f_node, f_edge, f_tri)
    
        '''
        training
        '''
        # save the model
        model_path = r'./model_nn_' + model_name
        '''for filter banks'''
        # model_path = r'./model_filter_' + model_name
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if model_name == 'sccnn_node':
            model_save_path = model_path+'/%dlayers_%dorder_%dfeatures_%drlz'%(args.layers,k,hidden_features,args.realizations)
            
        model = sccnn(F_in=1, F_intermediate=F_intermediate, F_out=hidden_features, b1=B1, b2=B2_train, l0=L0, l1l=L1l, l1u=L1u, l2=L2, d1=D1, d2=D2, d3=D3, d5=D5,k00=k00,k0p=k0p,k1n=k1n,k11=k11,k12=k12,k1p=k1p,k2n=k2n,k22=k22, sigma=sigma, model_name=model_name)
        model.load_state_dict(torch.load(model_save_path,map_location=device)['model'],strict=False)
        pred = MLPPredictor(h_feats=hidden_features) 
        pred.load_state_dict(torch.load(model_save_path,map_location=device)['pred'],strict=False)
        model.to(device)
        pred.to(device)
            
        h,h_edges,h_tri = model(f_input)
    
        for i in range(len(pert_list_2)):
            epsilon_2 = pert_list_2[i]
            pert_tri = np.mean([np.diag(np.random.rand(b2_train.shape[1])*epsilon_2-0.5*epsilon_2) for k in range(10)],0) 
            if epsilon_2 == 0:
                pert_tri = np.zeros_like(pert_tri)
            '''before'''
            # E2_rlz = np.mean([np.random.randn(L2.shape[0],L2.shape[1]) for k in range(10)],0)
            # E2 = torch.tensor(E2_rlz/la.norm(E2_rlz)*epsilon_2,dtype=torch.float32)
            # E2 = E2.to(device)
            # L2_pert = L2 + E2 @ L2 + L2 @ E2

            # E2_rlz = np.mean([np.random.randn(L1u.shape[0],L1u.shape[1]) for k in range(10)],0)
            # E2 = torch.tensor(E2_rlz/la.norm(E2_rlz)*epsilon_2,dtype=torch.float32)
            # E2 = E2.to(device)
            # L1u_pert = L1 + E2 @ L1u + L1u @ E2

            '''L1'''
            D2 = np.maximum(np.diag(abs(b2_train)@np.ones(b2_train.shape[1])),np.identity(b2_train.shape[0]))
            D3 = 1/3*np.identity(b2_train.shape[1])
            D3_pert = D3 + pert_tri 

            L1u_pert = la.inv(sla.fractional_matrix_power(D2,0.5)) @ b2_train @ D3_pert @ b2_train.T @ la.inv(sla.fractional_matrix_power(D2,0.5))
            print(L1u_pert,L1u)
            print(D3_pert,D3)
            '''L2'''
            D5 = np.diag(abs(b2_train)@np.ones(b2_train.shape[1]))
            D4 = np.eye(b2_train.shape[1])
            D4_pert = np.eye(b2_train.shape[1]) + pert_tri
            L2_pert = sla.fractional_matrix_power(D4_pert,0.5)@b2_train.T @la.pinv(D5)@ b2_train@sla.fractional_matrix_power(D4_pert,0.5)

            pert_tri = torch.tensor(pert_tri,dtype=torch.float32,device=device)
            L1u_pert = torch.tensor(L1u_pert,dtype=torch.float32,device=device)
            L2_pert = torch.tensor(L2_pert,dtype=torch.float32,device=device)

            D2 = torch.tensor(D2,dtype=torch.float32,device=device)    
            D3 = torch.tensor(D3,dtype=torch.float32,device=device)    
            D3_pert = torch.tensor(D3_pert,dtype=torch.float32,device=device)  
            D4 = torch.tensor(D4,dtype=torch.float32,device=device)    
            D4_pert = torch.tensor(D4_pert,dtype=torch.float32,device=device)    
            D5 = torch.tensor(D5,dtype=torch.float32,device=device)
            
            '''measure the influence of pert edge on three laplacians and projection matrices '''
            epsilon_rlt_1u[i,j] = tla.norm(L1u_pert-L1u,2)/tla.norm(L1u,2)
            epsilon_rlt_2[i,j] = tla.norm(L2_pert-L2,2)/tla.norm(L2,2)
            print('pert laps',epsilon_2,epsilon_rlt_1u[i,j],epsilon_rlt_2[i,j])

            proj_21 = B2_train @ D3
            proj_21_pert = B2_train @ D3_pert
            proj_12 = D4 @ B2_train.T @ torch.pinverse(D5) 
            proj_12_pert = D4_pert @ B2_train.T @ torch.pinverse(D5)
            epsilon_21[i,j] = tla.norm(proj_21_pert - proj_21,2)/tla.norm(proj_21,2)
            epsilon_12[i,j] = tla.norm(proj_12_pert - proj_12,2)/tla.norm(proj_12,2) 
            print('pert projs',epsilon_21[i,j], epsilon_12[i,j])

            model_pert = sccnn_stability(F_in=1, F_intermediate=F_intermediate, F_out=hidden_features, b1=B1, b2=B2_train, l0=L0, l1l=L1l, l1u=L1u_pert, l2=L2_pert, d1=D1, d2=D2, d3=D3_pert, d4=D4_pert, d5=D5,k00=k00,k0p=k0p,k1n=k1n,k11=k11,k12=k12,k1p=k1p,k2n=k2n,k22=k22, sigma=sigma, model_name=model_name+'_stability')
            
            model_pert.to(device)
            model_pert.load_state_dict(torch.load(model_save_path,map_location=device)['model'],strict=False)

            with torch.no_grad():
                fi_test_pos, fj_test_pos, fk_test_pos = h[test_pos_i], h[test_pos_j], h[test_pos_k]
                test_pos_score = pred(fi_test_pos, fj_test_pos, fk_test_pos)
                fi_test_neg, fj_test_neg, fk_test_neg = h[test_neg_i], h[test_neg_j], h[test_neg_k]
                test_neg_score = pred(fi_test_neg, fj_test_neg, fk_test_neg)
                
                h_pert,h_edges_pert,h_tri_pert = model_pert(f_input)
                
                dist[i,j] = (torch.linalg.norm(h_pert-h)/torch.linalg.norm(h))
                dist_edge[i,j] = (torch.linalg.norm(h_edges_pert-h_edges)/torch.linalg.norm(h_edges))
                dist_tri[i,j] = (torch.linalg.norm(h_tri_pert-h_tri)/torch.linalg.norm(h_tri))
                print( dist[i,j], dist_edge[i,j], dist_tri[i,j])
                
                fi_test_pos_pert, fj_test_pos_pert, fk_test_pos_pert = h_pert[test_pos_i], h_pert[test_pos_j], h_pert[test_pos_k]
                test_pos_score_pert = pred(fi_test_pos_pert, fj_test_pos_pert, fk_test_pos_pert)

                fi_test_neg_pert, fj_test_neg_pert, fk_test_neg_pert = h_pert[test_neg_i], h_pert[test_neg_j], h_pert[test_neg_k]
                test_neg_score_pert = pred(fi_test_neg_pert, fj_test_neg_pert, fk_test_neg_pert)
                # compute the auc
                auc[i,j] = (compute_auc(test_pos_score, test_neg_score))
                auc_pert[i,j] = (compute_auc(test_pos_score_pert, test_neg_score_pert))
                print(auc[i,j],auc_pert[i,j])
            
    var_path = r'./pert_var_nn_' + model_name +'_pert_2'
    # var_path = r'./pert_var_filter_' + model_name +'_pert_2'

    if not os.path.exists(var_path):
        os.makedirs(var_path)
  
    if model_name == 'sccnn_node':
        var_save_path = var_path+'/%dlayers_%dorders_%dfeatures.npy'%(args.layers,k,hidden_features)

    
    with open(var_save_path, 'wb') as f:
        np.save(f, auc)
        np.save(f, auc_pert)
        np.save(f, dist)    
        np.save(f, dist_edge) 
        np.save(f, dist_tri)   
        np.save(f, epsilon_2)
        np.save(f, epsilon_rlt_1u)
        np.save(f, epsilon_rlt_2)
        np.save(f, epsilon_12) 
        np.save(f, epsilon_21)  

if __name__ == "__main__":
    main()
