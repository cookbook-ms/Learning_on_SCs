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
from scnn_einsum import scnn
# for filters
#from scnn_einsum_id import scnn
from tetra_predictor import compute_auc, compute_loss, MLPPredictor


def main():
    '''
    hyperparameter
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-hd", "--hidden_features", help="the number of hidden features", type=int, default=16)
    parser.add_argument("-m","--model_name",help="the model name", type=str, default='gnn')
    parser.add_argument("-l", "--layers", help="the number of layers", type=int, default=1)
    parser.add_argument("-k","--filter_order_snn",help="the filter order of the model snn", type=int, default=3)
    parser.add_argument("-k1","--filter_order_scnn_k1",help="the filter order of the model scnn", type=int, default=3)
    parser.add_argument("-k2","--filter_order_scnn_k2",help="the filter order of the model scnn", type=int, default=5)
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
    topdim = 5
    boundaries = np.load('{}/{}_boundaries.npy'.format(prefix,starting_node),allow_pickle=True)
    boundaries = [boundaries[i].toarray() for i in range(topdim+1)]
    
    test_perc = 0.10
    val_perc = 0.10
    # triangle-to-tetra incidence matrix
    b3 = boundaries[2]
    b2 = boundaries[1]
    print(np.nonzero(b3[:,0 ]))
    b1 = boundaries[0]
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
    #print(node_idx[pos_edge_1][:],node_idx[pos_edge_2][:],node_idx[pos_edge_3][:],node_idx[pos_edge_4][:],node_idx[pos_edge_5][:],node_idx[pos_edge_6][:])
    pos_node_1,pos_node_2,pos_node_3,pos_node_4 = node_idx[pos_edge_1][:,0],node_idx[pos_edge_1][:,1],node_idx[pos_edge_6][:,0],node_idx[pos_edge_6][:,1] 
    print(pos_node_1,pos_node_2,pos_node_3,pos_node_4)
    
    neg_i, neg_j, neg_k, neg_l  = tri_idx[neg_idx][:,0], tri_idx[neg_idx][:,1], tri_idx[neg_idx][:,2], tri_idx[neg_idx][:,3]
    # from triangle index looking for the edge index
    neg_edge_1,neg_edge_2,neg_edge_3,neg_edge_4,neg_edge_5,neg_edge_6 = edge_idx[neg_i][:,0],edge_idx[neg_i][:,1],edge_idx[neg_i][:,2],edge_idx[neg_j][:,1],edge_idx[neg_j][:,2],edge_idx[neg_k][:,2]
    #print(node_idx[neg_edge_1][:],node_idx[neg_edge_2][:],node_idx[neg_edge_3][:],node_idx[neg_edge_4][:],node_idx[neg_edge_5][:],node_idx[neg_edge_6][:])
    neg_node_1,neg_node_2,neg_node_3,neg_node_4 = node_idx[neg_edge_1][:,0],node_idx[neg_edge_1][:,1],node_idx[neg_edge_6][:,0],node_idx[neg_edge_6][:,1] 
    print(neg_node_1,neg_node_2,neg_node_3,neg_node_4)
    # split the traing-testing data for both positive and negative samples and obtain the indices of the positive and negative, training and testing triangles 
    test_size_pos = int(pos_size * test_perc)
    train_size_pos = pos_size - test_size_pos
    val_size_pos = int(pos_size * val_perc)
    train_size_pos = train_size_pos - val_size_pos
    
    train_pos_i, train_pos_j, train_pos_k, train_pos_l = pos_node_1[:train_size_pos], pos_node_2[:train_size_pos], pos_node_3[:train_size_pos], pos_node_4[:train_size_pos]
    
    val_pos_i, val_pos_j, val_pos_k, val_pos_l = pos_node_1[train_size_pos:(train_size_pos+val_size_pos)], pos_node_2[train_size_pos:(train_size_pos+val_size_pos)], pos_node_3[train_size_pos:(train_size_pos+val_size_pos)], pos_node_4[train_size_pos:(train_size_pos+val_size_pos)]
    
    test_pos_i, test_pos_j, test_pos_k, test_pos_l = pos_node_1[(train_size_pos+val_size_pos):], pos_node_2[(train_size_pos+val_size_pos):], pos_node_3[(train_size_pos+val_size_pos):], pos_node_4[(train_size_pos+val_size_pos):] 
    print(train_pos_i, train_pos_j, train_pos_k, train_pos_l)

    test_size_neg = int(neg_size * test_perc)
    train_size_neg = neg_size - test_size_neg
    val_size_neg = int(neg_size * val_perc)
    train_size_neg = train_size_neg - val_size_neg
    
    train_neg_i, train_neg_j, train_neg_k, train_neg_l = neg_node_1[:train_size_neg], neg_node_2[:train_size_neg], neg_node_3[:train_size_neg], neg_node_4[:train_size_neg]
    
    val_neg_i, val_neg_j, val_neg_k, val_neg_l = neg_node_1[train_size_neg:(train_size_neg+val_size_neg)], neg_node_2[train_size_neg:(train_size_neg+val_size_neg)], neg_node_3[train_size_neg:(train_size_neg+val_size_neg)], neg_node_4[train_size_neg:(train_size_neg+val_size_neg)]
    
    test_neg_i, test_neg_j, test_neg_k, test_neg_l = neg_node_1[(train_size_neg+val_size_neg):], neg_node_2[(train_size_neg+val_size_neg):], neg_node_3[(train_size_neg+val_size_neg):], neg_node_4[(train_size_neg+val_size_neg):] 
    
    print(train_pos_i, train_pos_j, train_pos_k, train_pos_l)
    print(' #triangles of training positive tetras:', len(train_pos_i), len(train_pos_j), len(train_pos_k), len(train_pos_l), '\n','#triangles of training negative tetras:', len(train_neg_i), len(train_neg_j), len(train_neg_k), len(train_neg_l), '\n',
    '#triangles of testing positive tetras:', len(test_pos_i), len(test_pos_j), len(test_pos_k), len(test_pos_l), '\n','#triangles of testing negative tetras:', len(test_neg_i), len(test_neg_j), len(test_neg_k), len(test_neg_l))
    
    '''
    build the incidence matrix used for training
    1. remove the negative examples in the incidence matrix b3, only keep the positive ones 
    2. remove the test_pos examples in the incidence matrix b3 and build the simplicial complex, or equivalently, only take the train_pos triangle examples 
    '''
    b1_train = b1
    '''normalized laplacian based on random walk'''
    D2 = np.maximum(np.diag(abs(b1_train)@np.ones(b1_train.shape[1])),np.identity(b1_train.shape[0]))
    L0 =  la.inv(sla.fractional_matrix_power(D2,0.5)) @ b1_train @ b1_train.T @ la.inv(sla.fractional_matrix_power(D2,0.5)) 
    norms = np.linalg.norm(L0)
    L0 = torch.tensor(L0,dtype=torch.float32,device=device)
    
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
        
    model = scnn(F_in=1, F_intermediate=F_intermediate, F_out=hidden_features, K=K, K1=K1, K2=K2, laplacian=L0, laplacian_l=None, laplacian_u=None, sigma=sigma, model_name=model_name)
    pred = MLPPredictor(h_feats=hidden_features) # the number of the features of the output of the scnn model 
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

    
    # log the loss and auc results
    if model_name =='gnn':
        losslogf = open("./%s_%dlayers_%dorders_%dfeatures_%drlz.txt" %(model_name,args.layers,K,hidden_features,args.realizations),"w")

 
    best_val = 1e6
    '''
    training
    '''
    # save the model
    model_path = r'./model_nn_' + model_name
    #model_path = r'./model_filter_' + model_name
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if model_name == 'gnn':
        model_save_path = model_path+'/%dlayers_%dorders_%dfeatures_%drlz'%(args.layers,K,hidden_features,args.realizations)  
    # consider number of 100 epochs 
    for e in range(args.epochs):
        '''
        forward
        ''' 
        h = model(f_node)
        '''
        here we need to preprocess, i.e., reorganizing the matrix of edge features h for prediction input, in the form of, e.g., train_pos_i, train_pos_, train_pos_k, for each triangle, 
        '''
        fi_train_pos, fj_train_pos, fk_train_pos, fl_train_pos = h[train_pos_i], h[train_pos_j], h[train_pos_k], h[train_pos_l]
        train_pos_score = pred(fi_train_pos, fj_train_pos, fk_train_pos, fl_train_pos)

        fi_train_neg, fj_train_neg, fk_train_neg, fl_train_neg = h[train_neg_i], h[train_neg_j], h[train_neg_k], h[train_neg_l]
        train_neg_score = pred(fi_train_neg, fj_train_neg, fk_train_neg, fl_train_neg)

        labels = torch.cat([torch.ones(train_pos_score.shape[0]), torch.zeros(train_neg_score.shape[0])]).to(device)
        '''compute the loss'''
        loss = compute_loss(train_pos_score, train_neg_score, labels)
        
        losslogf.write("epoch %d, loss: %f\n" %(e, loss.item()))
        losslogf.flush()

        '''val'''
        with torch.no_grad():
            fi_val_pos, fj_val_pos, fk_val_pos, fl_val_pos = h[val_pos_i], h[val_pos_j], h[val_pos_k], h[val_pos_l]
            val_pos_score = pred(fi_val_pos, fj_val_pos, fk_val_pos,fl_val_pos)

            fi_val_neg, fj_val_neg, fk_val_neg, fl_val_neg = h[val_neg_i], h[val_neg_j], h[val_neg_k], h[val_neg_l]
            val_neg_score = pred(fi_val_neg, fj_val_neg, fk_val_neg,fl_val_neg)
            vlabels = torch.cat([torch.ones(val_pos_score.shape[0]), torch.zeros(val_neg_score.shape[0])]).to(device)
            vloss = compute_loss(val_pos_score, val_neg_score, vlabels)
            vauc = compute_auc(val_pos_score, val_neg_score)
            
            fi_test_pos, fj_test_pos, fk_test_pos, fl_test_pos = h[test_pos_i], h[test_pos_j], h[test_pos_k], h[test_pos_l]
            test_pos_score = pred(fi_test_pos, fj_test_pos, fk_test_pos, fl_test_pos)

            fi_test_neg, fj_test_neg, fk_test_neg, fl_test_neg = h[test_neg_i], h[test_neg_j], h[test_neg_k], h[test_neg_l]
            test_neg_score = pred(fi_test_neg, fj_test_neg, fk_test_neg, fl_test_neg)
            tlabels = torch.cat([torch.ones(test_pos_score.shape[0]), torch.zeros(test_neg_score.shape[0])]).to(device)
            test_loss = compute_loss(test_pos_score, test_neg_score, tlabels)
            test_auc = compute_auc(test_pos_score, test_neg_score)

            if vloss < best_val:
                best_val = vloss
                print('curent epoch:', e)
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


    # model = scnn(F_in=1, F_intermediate=F_intermediate, F_out=hidden_features, K=K, K1=K1, K2=K2, laplacian=L0, laplacian_l=None, laplacian_u=None, sigma=sigma, model_name=model_name)
    # pred = MLPPredictor(h_feats=hidden_features) # the number of the features of the output of the scnn model 
    # model.to(device)
    # pred.to(device)
    
    model.load_state_dict(torch.load(model_save_path,map_location=device)['model'],strict=False)
    pred.load_state_dict(torch.load(model_save_path,map_location=device)['pred'],strict=False)
    h = model(f_node)
    with torch.no_grad():
        fi_val_pos, fj_val_pos, fk_val_pos, fl_val_pos = h[val_pos_i], h[val_pos_j], h[val_pos_k], h[val_pos_l]
        val_pos_score = pred(fi_val_pos, fj_val_pos, fk_val_pos, fl_val_pos)

        fi_val_neg, fj_val_neg, fk_val_neg, fl_val_neg = h[val_neg_i], h[val_neg_j], h[val_neg_k], h[val_neg_l]
        val_neg_score = pred(fi_val_neg, fj_val_neg, fk_val_neg, fl_val_neg)
        vlabels = torch.cat([torch.ones(val_pos_score.shape[0]), torch.zeros(val_neg_score.shape[0])]).to(device)
        vloss = compute_loss(val_pos_score, val_neg_score, vlabels)
        v_auc = compute_auc(val_pos_score, val_neg_score)
        
        fi_test_pos, fj_test_pos, fk_test_pos, fl_test_pos = h[test_pos_i], h[test_pos_j], h[test_pos_k], h[test_pos_l]
        test_pos_score = pred(fi_test_pos, fj_test_pos, fk_test_pos, fl_test_pos)

        fi_test_neg, fj_test_neg, fk_test_neg, fl_test_neg = h[test_neg_i], h[test_neg_j], h[test_neg_k], h[test_neg_l]
        test_neg_score = pred(fi_test_neg, fj_test_neg, fk_test_neg, fl_test_neg)
        tlabels = torch.cat([torch.ones(test_pos_score.shape[0]), torch.zeros(test_neg_score.shape[0])]).to(device)
        test_loss = compute_loss(test_pos_score, test_neg_score, tlabels)
        tes_auc = compute_auc(test_pos_score, test_neg_score)
        losslogf.write("AUC: %f\n" %(tes_auc))
        losslogf.flush()
        print('val auc, test AUC',v_auc,tes_auc)
        print('val loss, test loss',vloss,test_loss)


    losslogf.close()


if __name__ == "__main__":
    main()