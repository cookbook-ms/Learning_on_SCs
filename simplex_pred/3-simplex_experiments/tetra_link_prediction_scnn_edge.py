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
from scnn_einsum_id import scnn
from tetra_predictor import compute_auc, compute_loss, MLPPredictor_edge


def main():
    '''
    hyperparameter
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-hd", "--hidden_features", help="the number of hidden features", type=int, default=16)
    parser.add_argument("-m","--model_name",help="the model name", type=str, default='scnn_edge')
    parser.add_argument("-l", "--layers", help="the number of layers", type=int, default=4)
    parser.add_argument("-k","--filter_order_snn",help="the filter order of the model snn", type=int, default=2)
    parser.add_argument("-k1","--filter_order_scnn_k1",help="the filter order of the model scnn", type=int, default=3)
    parser.add_argument("-k2","--filter_order_scnn_k2",help="the filter order of the model scnn", type=int, default=3)
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
    
    L1 = torch.tensor(L1,dtype=torch.float32,device=device)
    L1l = torch.tensor(L1l,dtype=torch.float32,device=device)
    L1u = torch.tensor(L1u,dtype=torch.float32,device=device)
    
    '''
    Define the feature learning model and predictor;
    Define the optimizer
    Define the input features -- the edge flow 
    training step
    '''
    sigma = activations[args.activations]
    hidden_features = args.hidden_features
    K = args.filter_order_snn
    K1 = args.filter_order_scnn_k1
    K2 = args.filter_order_scnn_k2
    model_name = args.model_name
    print(model_name)

    F_intermediate = []
    for i in range(args.layers):
        F_intermediate.append(hidden_features)
        
    model = scnn(F_in=1, F_intermediate=F_intermediate, F_out=hidden_features,  K=K, K1=K1, K2=K2, laplacian=L1, laplacian_l=L1l, laplacian_u=L1u, sigma=sigma, model_name=model_name)
    pred = MLPPredictor_edge(h_feats=hidden_features) # the number of the features of the output of the scnn model 
    model.to(device)
    pred.to(device)

    lr = args.learning_rate
    # define the loss and optimizer 
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr =lr) 

    # use the edge flow as the input feature to perform triangle prediction
    f_edge = torch.tensor(np.array(cochains[1]),dtype=torch.float32)
    f_edge.resize_(f_edge.shape[0],1)
    print(f_edge.dtype)
    f_edge = f_edge.to(device)
    
    # log the loss and auc results
    if model_name == 'snn':
        losslogf = open("./%s_edge_%dlayers_%dorders_%dfeatures_%drlz.txt" %(model_name,args.layers,K,hidden_features,args.realizations),"w")
    elif model_name == 'scnn':
        losslogf = open("./%s_edge_%dlayers_%d_%dorders_%dfeatures_%drlz.txt" %(model_name,args.layers,K1,K2,hidden_features,args.realizations),"w")
    elif model_name == 'psnn':
        losslogf = open("./%s_edge_%dlayers_%dfeatures_%drlz.txt" %(model_name,args.layers,hidden_features,args.realizations),"w")

    best_val = 1e6
    '''
    training
    '''
    # save the model
    model_path = r'./model_nn_' + model_name
    '''for filters'''
    model_path = r'./model_filter_' + model_name
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if model_name == 'snn':
        model_save_path = model_path+'/edge_%dlayers_%dorders_%dfeatures_%drlz'%(args.layers,K,hidden_features,args.realizations)     
    elif model_name == 'scnn':
         model_save_path = model_path+'/edge_%dlayers_%d_%dorders_%dfeatures_%drlz'%(args.layers,K1,K2,hidden_features,args.realizations)
    elif model_name == 'psnn':
        model_save_path = model_path+'/edge_%dlayers_%dfeatures_%drlz'%(args.layers,hidden_features,args.realizations)

    # consider number of 100 epochs 
    for e in range(args.epochs):
        '''
        forward
        ''' 
        h = model(f_edge)
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
    
    h = model(f_edge)  
      
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