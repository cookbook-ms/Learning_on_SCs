#!/usr/bin/env python3.8.5

import torch
import torch.nn as nn
import torch.nn.functional
import torch.utils.data as data
import numpy.linalg as la
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
import itertools
import sys
sys.path.append('.')
import argparse
import os
from pathlib import Path
path = Path(__file__).parent.absolute()
os.chdir(path)
from tetra_predictor import compute_auc, compute_loss, MLPPredictor_edge

from sccnn_model import sccnn

def main():
    '''
    hyperparameter
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-hd", "--hidden_features", help="the number of hidden features", type=int, default=16)
    parser.add_argument("-m","--model_name",help="the model name", type=str, default='sccnn_edge_sc_2')
    parser.add_argument("-l", "--layers", help="the number of layers", type=int, default=1)
    
    parser.add_argument("-k","--filter_order_same",help="the filter order of the model all filters", type=int, default=0)
    parser.add_argument("-k00","--filter_order_H00",help="the filter order of the model H00", type=int, default=1)
    parser.add_argument("-k0p","--filter_order_H0p",help="the filter order of the model H0n", type=int, default=1)
    parser.add_argument("-k1n","--filter_order_H1n",help="the filter order of H1p", type=int, default=1)
    parser.add_argument("-k11","--filter_order_H11_lower",help="the lower filter order of H11", type=int, default=1)
    parser.add_argument("-k12","--filter_order_H11_upper",help="the upper filter order of H11", type=int, default=1)
    parser.add_argument("-k1p","--filter_order_H1p",help="the filter order of H1p", type=int, default=1)
    parser.add_argument("-k2n","--filter_order_H2n",help="the filter order of the model H2n", type=int, default=1)
    parser.add_argument("-k21","--filter_order_H21",help="the filter order of the model H21", type=int, default=1)
    parser.add_argument("-k22","--filter_order_H22",help="the filter order of the model H22", type=int, default=1)
    parser.add_argument("-k2p","--filter_order_H2p",help="the filter order of the model H2p", type=int, default=1)
    parser.add_argument("-k3n","--filter_order_H3n",help="the filter order of the model H3n", type=int, default=1)
    parser.add_argument("-k33","--filter_order_H33",help="the filter order of the model H33", type=int, default=1)
    
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
    torch.manual_seed(1337*args.realizations)
    np.random.seed(1337*args.realizations)
    prefix = '../data/s2_3_collaboration_complex'
    starting_node = '150250' 
    
    test_perc = 0.10
    val_perc = 0.10
    
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    topdim = 5
    boundaries = np.load('{}/{}_boundaries.npy'.format(prefix,starting_node),allow_pickle=True)
    boundaries = [boundaries[i].toarray() for i in range(topdim+1)]
    # triangle-to-tetra incidence matrix
    b3 = boundaries[2]
    b2 = boundaries[1]
    b1 = boundaries[0]
    print(np.nonzero(b3[:,0 ]))
    # 3-simplex ids
    tetra_ids = np.arange(b3.shape[1])
    # the corresponding triangle ids of each tetra - the nonzero row indices are the triangle ids, stored in a matrix
    tri_idx = np.reshape(np.nonzero(b3.T)[1],(-1,4)) 
    edge_idx = np.reshape(np.nonzero(b2.T)[1],(-1,3)) 
    node_idx = np.reshape(np.nonzero(b1.T)[1],(-1,2))
    print(tri_idx[330])
    '''
    Data preprocessing
    1. use cochains[3], tetra signal, to split the tetras as positive and negative samples, i.e., closed and open ones, is this necessary or we can just consider the synthetic case where the split is random ?
     -- here we first consider the former choice 
    '''

    # the underlying true cochains
    cochains_dic = np.load('{}/{}_cochains.npy'.format(prefix,starting_node),allow_pickle=True)
    cochains =[list(cochains_dic[i].values()) for i in range(len(cochains_dic))]
    f_tetra = np.array(cochains[3])
    a = f_tetra<=7
    # neg_idx = np.where(a!=0)[0]
    # pos_idx = np.where(a==0)[0]
    neg_idx = np.random.permutation(np.where(a!=0)[0])
    pos_idx = np.random.permutation(np.where(a==0)[0])
    # take the edge cochain as the input feature for traignale prediction
    print(pos_idx, neg_idx)

    # tri_idx_perm = np.random.permutation(tri_idx)
    # print(tri_idx_perm)
    '''
    split the whole tetra set into positive and negative samples, i.e., closed and open tetra 
    - we use the triangle flow, i.e., the number of citations of papers with three authors, as the 
    '''
    pos_size = len(pos_idx)
    neg_size = len(neg_idx)
    print(pos_size, neg_size)
    pos_i, pos_j, pos_k, pos_l = tri_idx[pos_idx][:,0], tri_idx[pos_idx][:,1], tri_idx[pos_idx][:,2], tri_idx[pos_idx][:,3]
    # from triangle index looking for the edge index
    pos_edge_1,pos_edge_2,pos_edge_3,pos_edge_4,pos_edge_5,pos_edge_6 = edge_idx[pos_i][:,0],edge_idx[pos_i][:,1],edge_idx[pos_i][:,2],edge_idx[pos_j][:,1],edge_idx[pos_j][:,2],edge_idx[pos_k][:,2]
    
    neg_i, neg_j, neg_k, neg_l  = tri_idx[neg_idx][:,0], tri_idx[neg_idx][:,1], tri_idx[neg_idx][:,2], tri_idx[neg_idx][:,3]
    # from triangle index looking for the edge index
    neg_edge_1,neg_edge_2,neg_edge_3,neg_edge_4,neg_edge_5,neg_edge_6 = edge_idx[neg_i][:,0],edge_idx[neg_i][:,1],edge_idx[neg_i][:,2],edge_idx[neg_j][:,1],edge_idx[neg_j][:,2],edge_idx[neg_k][:,2]
    # split the traing-testing data for both positive and negative samples and obtain the indices of the positive and negative, training and testing triangles 
    test_size_pos = int(pos_size * test_perc)
    train_size_pos = pos_size - test_size_pos
    val_size_pos = int(pos_size * val_perc)
    train_size_pos = train_size_pos - val_size_pos

    train_pos_edge_1, train_pos_edge_2, train_pos_edge_3, train_pos_edge_4, train_pos_edge_5, train_pos_edge_6 = pos_edge_1[:train_size_pos],pos_edge_2[:train_size_pos],pos_edge_3[:train_size_pos],pos_edge_4[:train_size_pos],pos_edge_5[:train_size_pos],pos_edge_6[:train_size_pos]
    
    val_pos_edge_1, val_pos_edge_2, val_pos_edge_3, val_pos_edge_4, val_pos_edge_5, val_pos_edge_6 = pos_edge_1[train_size_pos:(train_size_pos+val_size_pos)],pos_edge_2[train_size_pos:(train_size_pos+val_size_pos)],pos_edge_3[train_size_pos:(train_size_pos+val_size_pos)],pos_edge_4[train_size_pos:(train_size_pos+val_size_pos)],pos_edge_5[train_size_pos:(train_size_pos+val_size_pos)],pos_edge_6[train_size_pos:(train_size_pos+val_size_pos)]
    
    test_pos_edge_1, test_pos_edge_2, test_pos_edge_3, test_pos_edge_4, test_pos_edge_5, test_pos_edge_6 = pos_edge_1[(train_size_pos+val_size_pos):],pos_edge_2[(train_size_pos+val_size_pos):],pos_edge_3[(train_size_pos+val_size_pos):],pos_edge_4[(train_size_pos+val_size_pos):],pos_edge_5[(train_size_pos+val_size_pos):],pos_edge_6[(train_size_pos+val_size_pos):]


    test_size_neg = int(neg_size * test_perc)
    train_size_neg = neg_size - test_size_neg
    val_size_neg = int(neg_size * val_perc)
    train_size_neg = train_size_neg - val_size_neg
    
    train_neg_edge_1, train_neg_edge_2, train_neg_edge_3, train_neg_edge_4, train_neg_edge_5, train_neg_edge_6 = neg_edge_1[:train_size_neg],neg_edge_2[:train_size_neg],neg_edge_3[:train_size_neg],neg_edge_4[:train_size_neg],neg_edge_5[:train_size_neg],neg_edge_6[:train_size_neg]
    
    val_neg_edge_1, val_neg_edge_2, val_neg_edge_3, val_neg_edge_4, val_neg_edge_5, val_neg_edge_6 = neg_edge_1[train_size_neg:(train_size_neg+val_size_neg)],neg_edge_2[train_size_neg:(train_size_neg+val_size_neg)],neg_edge_3[train_size_neg:(train_size_neg+val_size_neg)],neg_edge_4[train_size_neg:(train_size_neg+val_size_neg)],neg_edge_5[train_size_neg:(train_size_neg+val_size_neg)],neg_edge_6[train_size_neg:(train_size_neg+val_size_neg)]
    
    test_neg_edge_1, test_neg_edge_2, test_neg_edge_3, test_neg_edge_4, test_neg_edge_5, test_neg_edge_6 = neg_edge_1[(train_size_neg+val_size_neg):],neg_edge_2[(train_size_neg+val_size_neg):],neg_edge_3[(train_size_neg+val_size_neg):],neg_edge_4[(train_size_neg+val_size_neg):],neg_edge_5[(train_size_neg+val_size_neg):],neg_edge_6[(train_size_neg+val_size_neg):]
    
    '''
    build the incidence matrix used for training
    1. remove the negative examples in the incidence matrix b3, only keep the positive ones 
    2. remove the test_pos examples in the incidence matrix b3 and build the simplicial complex, or equivalently, only take the train_pos triangle examples 
    '''
    b3_train = b3[:,pos_idx][:,:train_size_pos]
    # Check if the training incidece matrix contains open triangles 
    (~b3_train.any(axis=0)).any() 
    l2u = b3_train@b3_train.T # the true one: laplacians_up[1].toarray()
    # edge-to-triangle incidence matrix 

    print(b2.shape)

    '''normalized laplacian based on random walk'''
    '''L2'''
    D2 = np.maximum(np.diag(abs(b3_train)@np.ones(b3_train.shape[1])),np.identity(b3_train.shape[0]))
    D1 = 3 * np.diag(abs(b2)@D2@np.ones(D2.shape[0]))
    D3 = 1/4*np.identity(b3_train.shape[1])
    if args.model_name == 'sccnn_edge_sc_2':
        D1 = np.diag(abs(b2)@np.ones(b2.shape[1]))
        D2 = np.identity(b2.shape[1])
    L2l = sla.fractional_matrix_power(D2,0.5) @ b2.T @ la.pinv(D1) @ b2 @ sla.fractional_matrix_power(D2,0.5)
    L2u = la.inv(sla.fractional_matrix_power(D2,0.5)) @ b3_train @ D3 @ b3_train.T @ la.inv(sla.fractional_matrix_power(D2,0.5))
    L2 = L2l + L2u
    # print(L2l,L2u,L2)
    L2l = torch.tensor(L2l,dtype=torch.float32,device=device)
    L2u = torch.tensor(L2u,dtype=torch.float32,device=device)
    L2 = torch.tensor(L2,dtype=torch.float32,device=device)
    
    '''L0'''
    b1_train = b1
    D200 = np.maximum(np.diag(abs(b1_train)@np.ones(b1_train.shape[1])),np.identity(b1_train.shape[0]))
    L0 =  la.inv(sla.fractional_matrix_power(D200,0.5)) @ b1_train @ b1_train.T @ la.inv(sla.fractional_matrix_power(D200,0.5)) 
    
    '''L1'''
    b2_train = b2
    D20 = np.maximum(np.diag(abs(b2_train)@np.ones(b2_train.shape[1])),np.identity(b2_train.shape[0]))
    D10 = 2 * np.diag(abs(b1)@D20@np.ones(D20.shape[0]))
    D30 = 1/3*np.identity(b2_train.shape[1])
    # L1l = D2 @ b1.T @ la.pinv(D1) @ b1 
    # L1u = b2_train @ D3 @ b2_train.T @ la.inv(D2)
    L1l = sla.fractional_matrix_power(D20,0.5) @ b1.T @ la.pinv(D10) @ b1 @ sla.fractional_matrix_power(D20,0.5)
    L1u = la.inv(sla.fractional_matrix_power(D20,0.5)) @ b2_train @ D30 @ b2_train.T @ la.inv(sla.fractional_matrix_power(D20,0.5))
    L1 = L1l + L1u
    
    '''L3'''
    D5 = np.diag(abs(b3_train)@np.ones(b3_train.shape[1]))
    L3 = b3_train.T @la.pinv(D5)@ b3_train
    
    L0 = torch.tensor(L0,dtype=torch.float32,device=device)
    L1 = torch.tensor(L1,dtype=torch.float32,device=device)
    L1l = torch.tensor(L1l,dtype=torch.float32,device=device)
    L1u = torch.tensor(L1u,dtype=torch.float32,device=device)
    B1 = torch.tensor(b1,dtype=torch.float32,device=device)
    B2 = torch.tensor(b2,dtype=torch.float32,device=device)
    B3_train = torch.tensor(b3_train,dtype=torch.float32,device=device)
    L3 = torch.tensor(L3,dtype=torch.float32,device=device)   
    D1 = torch.tensor(D1,dtype=torch.float32,device=device)    
    D2 = torch.tensor(D2,dtype=torch.float32,device=device)    
    D3 = torch.tensor(D3,dtype=torch.float32,device=device)    
    D5 = torch.tensor(D5,dtype=torch.float32,device=device) 
    D10 = torch.tensor(D10,dtype=torch.float32,device=device)    
    D20 = torch.tensor(D20,dtype=torch.float32,device=device)    
    D30 = torch.tensor(D30,dtype=torch.float32,device=device)    

    '''
    Define the feature learning model and predictor;
    Define the optimizer
    Define the input features -- the edge flow 
    training step
    '''
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
    k21 = k
    k22 = k
    k2p = k
    k3n = k
    k33 = k
    model_name = args.model_name
    print(model_name)

    F_intermediate = []
    for i in range(args.layers):
        F_intermediate.append(hidden_features)
        
    model = sccnn(F_in=1, F_intermediate=F_intermediate, F_out=hidden_features, b1=B1, b2=B2, b3=B3_train, l0=L0, l1l=L1l, l1u=L1u, l2l=L2l, l2u=L2u, l3=L3, d1=D1,d2=D2,d3=D3,d5=D5, d10=D10, d20=D20, d30=D30, k00=k00, k0p=k0p,k1n=k1n,k11=k11,k12=k12,k1p=k1p, k2n=k2n,k21=k21,k22=k22,k2p=k2p,k3n=k3n,k33=k33,sigma=sigma, model_name=model_name)
    pred = MLPPredictor_edge(h_feats=hidden_features) # the number of the features of the output of the scnn model 
    model.to(device)
    pred.to(device)

    lr = args.learning_rate
    # define the loss and optimizer 
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr =lr) 

    # use the edge flow as the input feature to perform triangle prediction
    f_node = torch.tensor(np.array(cochains[0]),dtype=torch.float32)
    f_node.resize_(f_node.shape[0],1)
    print(f_node.dtype)
    f_node = f_node.to(device)
    
    f_edge = torch.tensor(np.array(cochains[1]),dtype=torch.float32)
    f_edge.resize_(f_edge.shape[0],1)
    print(f_edge.dtype)
    f_edge = f_edge.to(device)
    
    f_tri = torch.tensor(np.array(cochains[2]),dtype=torch.float32)
    f_tri.resize_(f_tri.shape[0],1)
    print(f_tri.dtype)
    f_tri = f_tri.to(device)
    
    f_tetra = torch.tensor(np.zeros(b3_train.shape[1]),dtype=torch.float32)
    f_tetra.resize_(f_tetra.shape[0],1)
    print(f_tetra.dtype)
    f_tetra = f_tetra.to(device)
    
    f_input = (f_node, f_edge, f_tri)
    
    # log the loss and auc results
    if model_name in ['sccnn_edge_sc_2']:
        losslogf = open("./%s_%dlayers_%dorder_%dfeatures_%drlz.txt" %(model_name,args.layers,k,hidden_features,args.realizations),"w")

    best_val = 1e6
    '''
    training
    '''
    # save the model
    model_path = r'./model_nn_' + model_name
    '''for filters'''
    #model_path = r'./model_filter_' + model_name
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if model_name in ['sccnn_edge_sc_2']:
        model_save_path = model_path+'/%dlayers_%dorder_%dfeatures_%drlz'%(args.layers,k,hidden_features,args.realizations)     

    # consider number of 100 epochs 
    for e in range(args.epochs):
        '''
        forward
        ''' 
        h_node,h,h_tri = model(f_input)
        '''
        here we need to preprocess, i.e., reorganizing the matrix of edge features h for prediction input, in the form of, e.g., train_pos_i, train_pos_, train_pos_k, for each triangle, 
        '''
        f_train_pos_1,f_train_pos_2,f_train_pos_3,f_train_pos_4,f_train_pos_5,f_train_pos_6  = h[train_pos_edge_1],  h[train_pos_edge_2], h[train_pos_edge_3], h[train_pos_edge_4], h[train_pos_edge_5], h[train_pos_edge_6]
        train_pos_score = pred(f_train_pos_1,f_train_pos_2,f_train_pos_3,f_train_pos_4,f_train_pos_5,f_train_pos_6)

        f_train_neg_1,f_train_neg_2,f_train_neg_3,f_train_neg_4,f_train_neg_5,f_train_neg_6  = h[train_neg_edge_1],  h[train_neg_edge_2], h[train_neg_edge_3], h[train_neg_edge_4], h[train_neg_edge_5], h[train_neg_edge_6]
        train_neg_score = pred(f_train_neg_1,f_train_neg_2,f_train_neg_3,f_train_neg_4,f_train_neg_5,f_train_neg_6)

        labels = torch.cat([torch.ones(train_pos_score.shape[0]), torch.zeros(train_neg_score.shape[0])]).to(device)
        
        '''compute the loss'''
        loss = compute_loss(train_pos_score, train_neg_score, labels)
        
        losslogf.write("epoch %d, loss: %f\n" %(e, loss.item()))
        losslogf.flush()
        
        '''val'''
        with torch.no_grad():
            f_val_pos_1,f_val_pos_2,f_val_pos_3,f_val_pos_4,f_val_pos_5,f_val_pos_6  = h[val_pos_edge_1],  h[val_pos_edge_2], h[val_pos_edge_3], h[val_pos_edge_4], h[val_pos_edge_5], h[val_pos_edge_6]
            val_pos_score = pred(f_val_pos_1,f_val_pos_2,f_val_pos_3,f_val_pos_4,f_val_pos_5,f_val_pos_6)

            f_val_neg_1,f_val_neg_2,f_val_neg_3,f_val_neg_4,f_val_neg_5,f_val_neg_6  = h[val_neg_edge_1],  h[val_neg_edge_2], h[val_neg_edge_3], h[val_neg_edge_4], h[val_neg_edge_5], h[val_neg_edge_6]
            val_neg_score = pred(f_val_neg_1,f_val_neg_2,f_val_neg_3,f_val_neg_4,f_val_neg_5,f_val_neg_6)
            
            vlabels = torch.cat([torch.ones(val_pos_score.shape[0]), torch.zeros(val_neg_score.shape[0])]).to(device)
            vloss = compute_loss(val_pos_score, val_neg_score, vlabels)
            vauc = compute_auc(val_pos_score, val_neg_score)
            
            f_test_pos_1,f_test_pos_2,f_test_pos_3,f_test_pos_4,f_test_pos_5,f_test_pos_6  = h[test_pos_edge_1],  h[test_pos_edge_2], h[test_pos_edge_3], h[test_pos_edge_4], h[test_pos_edge_5], h[test_pos_edge_6]
            test_pos_score = pred(f_test_pos_1,f_test_pos_2,f_test_pos_3,f_test_pos_4,f_test_pos_5,f_test_pos_6)

            f_test_neg_1,f_test_neg_2,f_test_neg_3,f_test_neg_4,f_test_neg_5,f_test_neg_6  = h[test_neg_edge_1],  h[test_neg_edge_2], h[test_neg_edge_3], h[test_neg_edge_4], h[test_neg_edge_5], h[test_neg_edge_6]
            test_neg_score = pred(f_test_neg_1,f_test_neg_2,f_test_neg_3,f_test_neg_4,f_test_neg_5,f_test_neg_6)
            
            tlabels = torch.cat([torch.ones(test_pos_score.shape[0]), torch.zeros(test_neg_score.shape[0])]).to(device)
            test_loss = compute_loss(test_pos_score, test_neg_score, tlabels)
            test_auc = compute_auc(test_pos_score, test_neg_score)

            if vloss < best_val:
                best_val = vloss
                print('current epoch:', e)
                torch.save({'model': model.state_dict(),'pred':pred.state_dict()},model_save_path)
                losslogf.write("model updated at epoch %d \n" %(e))
            print('epoch {},\n train loss: {}, val loss: {}, \n val auc {},  test auc {}'.format(e, loss, vloss, vauc, test_auc))    
            
        losslogf.write("epoch %d, \n train loss: %f, val loss: %f \n val auc: %f,  test auc: %f \n" %(e, loss.item(), vloss, vauc, test_auc))
        losslogf.flush()
            
        '''
        backward
        '''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.load_state_dict(torch.load(model_save_path,map_location=device)['model'],strict=False)
    pred.load_state_dict(torch.load(model_save_path,map_location=device)['pred'],strict=False)
    
    h_node,h,h_tri = model(f_input)
      
    with torch.no_grad():
        f_val_pos_1,f_val_pos_2,f_val_pos_3,f_val_pos_4,f_val_pos_5,f_val_pos_6  = h[val_pos_edge_1],  h[val_pos_edge_2], h[val_pos_edge_3], h[val_pos_edge_4], h[val_pos_edge_5], h[val_pos_edge_6]
        val_pos_score = pred(f_val_pos_1,f_val_pos_2,f_val_pos_3,f_val_pos_4,f_val_pos_5,f_val_pos_6)

        f_val_neg_1,f_val_neg_2,f_val_neg_3,f_val_neg_4,f_val_neg_5,f_val_neg_6  = h[val_neg_edge_1],  h[val_neg_edge_2], h[val_neg_edge_3], h[val_neg_edge_4], h[val_neg_edge_5], h[val_neg_edge_6]
        val_neg_score = pred(f_val_neg_1,f_val_neg_2,f_val_neg_3,f_val_neg_4,f_val_neg_5,f_val_neg_6)
            
        vlabels = torch.cat([torch.ones(val_pos_score.shape[0]), torch.zeros(val_neg_score.shape[0])]).to(device)
        vloss = compute_loss(val_pos_score, val_neg_score, vlabels)
        v_auc = compute_auc(val_pos_score, val_neg_score)
            
        f_test_pos_1,f_test_pos_2,f_test_pos_3,f_test_pos_4,f_test_pos_5,f_test_pos_6  = h[test_pos_edge_1],  h[test_pos_edge_2], h[test_pos_edge_3], h[test_pos_edge_4], h[test_pos_edge_5], h[test_pos_edge_6]
        test_pos_score = pred(f_test_pos_1,f_test_pos_2,f_test_pos_3,f_test_pos_4,f_test_pos_5,f_test_pos_6)

        f_test_neg_1,f_test_neg_2,f_test_neg_3,f_test_neg_4,f_test_neg_5,f_test_neg_6  = h[test_neg_edge_1],  h[test_neg_edge_2], h[test_neg_edge_3], h[test_neg_edge_4], h[test_neg_edge_5], h[test_neg_edge_6]
        test_neg_score = pred(f_test_neg_1,f_test_neg_2,f_test_neg_3,f_test_neg_4,f_test_neg_5,f_test_neg_6)
        
        tlabels = torch.cat([torch.ones(test_pos_score.shape[0]), torch.zeros(test_neg_score.shape[0])]).to(device)
        test_loss = compute_loss(test_pos_score, test_neg_score, tlabels)
        test_auc = compute_auc(test_pos_score, test_neg_score)
        losslogf.write("AUC: %f\n" %(test_auc))
        losslogf.flush()
        print('val loss, test loss',vloss,test_loss)
        print('val auc, test AUC',v_auc,test_auc)


    losslogf.close()

if __name__ == "__main__":
    main()