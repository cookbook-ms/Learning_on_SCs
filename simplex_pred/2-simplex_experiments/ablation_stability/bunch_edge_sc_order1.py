#!/usr/bin/env python3.8.5
'''
this is the script to implement bunch-edge when sc is of order 1
'''
import torch
import torch.nn as nn
import torch.nn.functional
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
from pathlib import Path
path = Path(__file__).parent.absolute()
os.chdir(path)
from tri_predictor import compute_auc, compute_loss, MLPPredictor
from bunch_model import bunch

def main():
    '''
    hyperparameter
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-hd", "--hidden_features", help="the number of hidden features", type=int, default=16)
    parser.add_argument("-m","--model_name",help="the model name", type=str, default='bunch_edge_sc_1')
    parser.add_argument("-l", "--layers", help="the number of layers", type=int, default=1) # note that here it is the number of intermediate layers
    parser.add_argument("-k","--filter_order_snn",help="the filter order of the model snn", type=int, default=3)
    parser.add_argument("-k1","--filter_order_scnn_k1",help="the filter order of the model scnn", type=int, default=2)
    parser.add_argument("-k2","--filter_order_scnn_k2",help="the filter order of the model scnn", type=int, default=4)
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

    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    print(device)
    topdim = 5
    boundaries = np.load('{}/{}_boundaries.npy'.format(prefix,starting_node),allow_pickle=True)

    boundaries = [boundaries[i].toarray() for i in range(topdim+1)]
    test_perc = 0.10
    val_perc = 0.10
    # edge-to-triangle incidence matrix 
    b2 = boundaries[1]
    print(b2.shape)
    # triangle ids
    tri_idx = np.arange(b2.shape[1])
    edge_idx = np.reshape(np.nonzero(b2.T)[1],(-1,3)) 
    print(edge_idx)
    print(np.nonzero(b2.T))
    '''
    Data preprocessing
    '''
    # the underlying true cochains
    cochains_dic = np.load('{}/{}_cochains.npy'.format(prefix,starting_node),allow_pickle=True)
    cochains =[list(cochains_dic[i].values()) for i in range(len(cochains_dic))]
    f_tri = np.array(cochains[2])
    a = f_tri<=7
    #neg_idx = np.where(a!=0)[0]
    #pos_idx = np.where(a==0)[0]
    neg_idx = np.random.permutation(np.where(a!=0)[0])
    pos_idx = np.random.permutation(np.where(a==0)[0])
    # take the edge cochain as the input feature for traignale prediction
    print(pos_idx, neg_idx)

    # tri_idx_perm = np.random.permutation(tri_idx)
    # print(tri_idx_perm)
    '''
    split the whole triangle set into positive and negative samples, i.e., closed and open triangles 
    - we use the triangle flow, i.e., the number of citations of papers with three authors, as the 
    '''
    pos_size = len(pos_idx)
    neg_size = len(neg_idx)
    print(pos_size, neg_size)
    pos_i, pos_j, pos_k = edge_idx[pos_idx][:,0], edge_idx[pos_idx][:,1], edge_idx[pos_idx][:,2]
    print(pos_i, pos_j, pos_k)
    neg_i, neg_j, neg_k = edge_idx[neg_idx][:,0], edge_idx[neg_idx][:,1], edge_idx[neg_idx][:,2]
    
    # split the traing-testing data for both positive and negative samples and obtain the indices of the positive and negative, training and testing edges 
    test_size_pos = int(pos_size * test_perc)
    train_size_pos = pos_size - test_size_pos
    val_size_pos = int(pos_size * val_perc)
    train_size_pos = train_size_pos - val_size_pos
    
    train_pos_i, train_pos_j, train_pos_k = pos_i[:train_size_pos], pos_j[:train_size_pos], pos_k[:train_size_pos]
    val_pos_i, val_pos_j, val_pos_k = pos_i[train_size_pos:(train_size_pos+val_size_pos)], pos_j[train_size_pos:(train_size_pos+val_size_pos)], pos_k[train_size_pos:(train_size_pos+val_size_pos)] 
    test_pos_i, test_pos_j, test_pos_k = pos_i[(train_size_pos+val_size_pos):], pos_j[(train_size_pos+val_size_pos):], pos_k[(train_size_pos+val_size_pos):] 
    print(train_pos_i, train_pos_j, train_pos_k)

    test_size_neg = int(neg_size * test_perc)
    train_size_neg = neg_size - test_size_neg
    val_size_neg = int(neg_size * val_perc)
    train_size_neg = train_size_neg - val_size_neg
    
    train_neg_i, train_neg_j, train_neg_k = neg_i[:train_size_neg], neg_j[:train_size_neg], neg_k[:train_size_neg]
    val_neg_i, val_neg_j, val_neg_k = neg_i[train_size_neg:(train_size_neg+val_size_neg)], neg_j[train_size_neg:(train_size_neg+val_size_neg)], neg_k[train_size_neg:(train_size_neg+val_size_neg)] 
    test_neg_i, test_neg_j, test_neg_k = neg_i[(train_size_neg+val_size_neg):], neg_j[(train_size_neg+val_size_neg):], neg_k[(train_size_neg+val_size_neg):] 
    print(train_neg_i, train_neg_j, train_neg_k)
    
    '''
    build the incidence matrix used for training
    1. remove the negative examples in the incidence matrix b2, only keep the positive ones 
    2. remove the test_pos examples in the incidence matrix b2 and build the simplicial complex, or equivalently, only take the train_pos triangle examples 
    '''
    b2_train = b2[:,pos_idx][:,:train_size_pos]
    # Check if the training incidece matrix contains open triangles 
    print((~b2_train.any(axis=0)).any()) 
    l1u = b2_train@b2_train.T 
    b1 = boundaries[0] 
    l1l = b1.T @ b1
    l1 = l1l + l1u
    '''L1'''
    D2 = np.maximum(np.diag(abs(b2_train)@np.ones(b2_train.shape[1])),np.identity(b2_train.shape[0]))
    D1 = 2 * np.diag(abs(b1)@D2@np.ones(D2.shape[0]))
    D3 = 1/3*np.identity(b2_train.shape[1])
    if args.model_name in ['bunch_edge_sc_1']:
        '''we need to redefine the normalization if there is no b2'''
        D1 = np.diag(abs(b1)@np.ones(b1.shape[1]))
        D2 = np.identity(b1.shape[1])
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
    
    model = bunch(F_in=1, F_intermediate=F_intermediate, F_out=hidden_features, b1=B1, b2=B2_train, l0=L0, l1=L1l, l2=L2, d1=D1, d2=D2, d3=D3, d5=D5, sigma=sigma, model_name=model_name)
    pred = MLPPredictor(h_feats=hidden_features) # the number of the features of the output of the scnn model 
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
    
    f_node = torch.tensor(np.array(cochains[0]),dtype=torch.float32)
    f_node.resize_(f_node.shape[0],1)
    print(f_node.dtype)
    f_node = f_node.to(device)
    
    f_tri = torch.tensor(np.zeros(b2_train.shape[1]),dtype=torch.float32)
    f_tri.resize_(f_tri.shape[0],1)
    f_tri = f_tri.to(device)
    
    f_input = (f_node, f_edge)
    print(type(f_input))
    # log the loss and auc results
    if model_name in ['bunch_edge_sc_1']:
        losslogf = open("./%s_%dlayers_%dfeatures_%drlz.txt" %(model_name,args.layers,hidden_features,args.realizations),"w")

    best_val = 1e6
    '''
    training
    '''
    # save the model
    model_path = r'./model_nn_' + model_name
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if model_name in ['bunch_edge_sc_1']:
        model_save_path = model_path+'/%dlayers_%dfeatures_%drlz'%(args.layers,hidden_features,args.realizations)
    # consider number of 100 epochs 
    for e in range(args.epochs):
        '''
        forward
        ''' 
        h_nodes,h = model(f_input)
        '''
        here we need to preprocess, i.e., reorganizing the matrix of edge features h for prediction input, in the form of, e.g., train_pos_i, train_pos_, train_pos_k, for each triangle, 
        '''
        fi_train_pos, fj_train_pos, fk_train_pos = h[train_pos_i], h[train_pos_j], h[train_pos_k]
        train_pos_score = pred(fi_train_pos, fj_train_pos, fk_train_pos)

        fi_train_neg, fj_train_neg, fk_train_neg = h[train_neg_i], h[train_neg_j], h[train_neg_k]
        train_neg_score = pred(fi_train_neg, fj_train_neg, fk_train_neg)

        labels = torch.cat([torch.ones(train_pos_score.shape[0]), torch.zeros(train_neg_score.shape[0])]).to(device)
        '''compute the loss'''
        loss = compute_loss(train_pos_score, train_neg_score, labels)

        losslogf.write("epoch %d, loss: %f\n" %(e, loss.item()))
        losslogf.flush()

        '''val'''
        with torch.no_grad():
            fi_val_pos, fj_val_pos, fk_val_pos = h[val_pos_i], h[val_pos_j], h[val_pos_k]
            val_pos_score = pred(fi_val_pos, fj_val_pos, fk_val_pos)

            fi_val_neg, fj_val_neg, fk_val_neg = h[val_neg_i], h[val_neg_j], h[val_neg_k]
            val_neg_score = pred(fi_val_neg, fj_val_neg, fk_val_neg)
            vlabels = torch.cat([torch.ones(val_pos_score.shape[0]), torch.zeros(val_neg_score.shape[0])]).to(device)
            vloss = compute_loss(val_pos_score, val_neg_score, vlabels)
            vauc = compute_auc(val_pos_score, val_neg_score)
                
            fi_test_pos, fj_test_pos, fk_test_pos = h[test_pos_i], h[test_pos_j], h[test_pos_k]
            test_pos_score = pred(fi_test_pos, fj_test_pos, fk_test_pos)

            fi_test_neg, fj_test_neg, fk_test_neg = h[test_neg_i], h[test_neg_j], h[test_neg_k]
            test_neg_score = pred(fi_test_neg, fj_test_neg, fk_test_neg)
            tlabels = torch.cat([torch.ones(test_pos_score.shape[0]), torch.zeros(test_neg_score.shape[0])]).to(device)
            test_loss = compute_loss(test_pos_score, test_neg_score, tlabels)
            test_auc = compute_auc(test_pos_score, test_neg_score)
            
            if vloss < best_val:
                best_val = vloss
                print('curent epoch:', e)
                torch.save({'model': model.state_dict(),'pred':pred.state_dict()},model_save_path)
                losslogf.write("model updated at epoch %d \n" %(e))
            print('epoch {},\n train loss: {}, val loss: {}, \n val auc {},  test auc {}'.format(e, loss, vloss, vauc, test_auc))    
            
        losslogf.write("epoch %d, \n train loss: %f, val loss: %f \n val auc: %f,  test auc: %f\n" %(e, loss.item(), vloss, vauc, test_auc))
        losslogf.flush()
        
        '''
        backward
        '''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        
         
    '''
    testing
    '''       
    
    model.load_state_dict(torch.load(model_save_path,map_location=device)['model'],strict=False)
    pred.load_state_dict(torch.load(model_save_path,map_location=device)['pred'],strict=False)
    
    h_nodes,h = model(f_input)

    with torch.no_grad():
        fi_val_pos, fj_val_pos, fk_val_pos = h[val_pos_i], h[val_pos_j], h[val_pos_k]
        val_pos_score = pred(fi_val_pos, fj_val_pos, fk_val_pos)

        fi_val_neg, fj_val_neg, fk_val_neg = h[val_neg_i], h[val_neg_j], h[val_neg_k]
        val_neg_score = pred(fi_val_neg, fj_val_neg, fk_val_neg)
        vlabels = torch.cat([torch.ones(val_pos_score.shape[0]), torch.zeros(val_neg_score.shape[0])]).to(device)
        vloss = compute_loss(val_pos_score, val_neg_score, vlabels)
        v_auc = compute_auc(val_pos_score, val_neg_score)
        
        fi_test_pos, fj_test_pos, fk_test_pos = h[test_pos_i], h[test_pos_j], h[test_pos_k]
        test_pos_score = pred(fi_test_pos, fj_test_pos, fk_test_pos)

        fi_test_neg, fj_test_neg, fk_test_neg = h[test_neg_i], h[test_neg_j], h[test_neg_k]
        test_neg_score = pred(fi_test_neg, fj_test_neg, fk_test_neg)
        tlabels = torch.cat([torch.ones(test_pos_score.shape[0]), torch.zeros(test_neg_score.shape[0])]).to(device)
        test_loss = compute_loss(test_pos_score, test_neg_score, tlabels)
        test_auc = compute_auc(test_pos_score, test_neg_score)
        losslogf.write("AUC: %f\n" %(test_auc))
        losslogf.flush()
        print('val auc, test AUC',v_auc,test_auc,test_loss,)
        print('+++',vloss)

    losslogf.close()


if __name__ == "__main__":
    main()