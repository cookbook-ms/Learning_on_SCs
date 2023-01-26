#!/usr/bin/env python3.8.5

import torch
import torch.nn as nn
import torch.nn.functional
import torch.utils.data as data
import numpy.linalg as la
import numpy as np
import scipy.sparse as sp
import itertools
import sys
sys.path.append('.')
import argparse
import sympy as sp
import os 
from pathlib import Path
path = Path(__file__).parent.absolute()
os.chdir(path)
from tetra_predictor import compute_auc, compute_loss, MLPPredictor

def main():
    harm_auc = []
    arith_auc = []
    geom_auc = []
    for i in range(10):
        torch.manual_seed(1337*i)
        np.random.seed(1337*i)

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
        print(np.nonzero(b3[:,0 ]))
        # 3-simplex ids
        tetra_ids = np.arange(b3.shape[1])
        # the corresponding triangle ids of each tetra - the nonzero row indices are the triangle ids, stored in a matrix
        tri_idx = np.reshape(np.nonzero(b3.T)[1],(-1,4)) 
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
        print(pos_i, pos_j, pos_k, pos_l)
        neg_i, neg_j, neg_k, neg_l  = tri_idx[neg_idx][:,0], tri_idx[neg_idx][:,1], tri_idx[neg_idx][:,2], tri_idx[neg_idx][:,3]
        print(neg_i, neg_j, neg_k, neg_l)
        # split the traing-testing data for both positive and negative samples and obtain the indices of the positive and negative, training and testing triangles 
        test_size_pos = int(pos_size * test_perc)
        train_size_pos = pos_size - test_size_pos
        val_size_pos = int(pos_size * val_perc)
        train_size_pos = train_size_pos - val_size_pos
        
        train_pos_i, train_pos_j, train_pos_k, train_pos_l = pos_i[:train_size_pos], pos_j[:train_size_pos], pos_k[:train_size_pos], pos_l[:train_size_pos]
        
        val_pos_i, val_pos_j, val_pos_k, val_pos_l = pos_i[train_size_pos:(train_size_pos+val_size_pos)], pos_j[train_size_pos:(train_size_pos+val_size_pos)], pos_k[train_size_pos:(train_size_pos+val_size_pos)], pos_l[train_size_pos:(train_size_pos+val_size_pos)]
        
        test_pos_i, test_pos_j, test_pos_k, test_pos_l = pos_i[(train_size_pos+val_size_pos):], pos_j[(train_size_pos+val_size_pos):], pos_k[(train_size_pos+val_size_pos):], pos_l[(train_size_pos+val_size_pos):] 


        test_size_neg = int(neg_size * test_perc)
        train_size_neg = neg_size - test_size_neg
        val_size_neg = int(neg_size * val_perc)
        train_size_neg = train_size_neg - val_size_neg
        
        train_neg_i, train_neg_j, train_neg_k, train_neg_l = neg_i[:train_size_neg], neg_j[:train_size_neg], neg_k[:train_size_neg], neg_l[:train_size_neg]
        
        val_neg_i, val_neg_j, val_neg_k, val_neg_l = neg_i[train_size_neg:(train_size_neg+val_size_neg)], neg_j[train_size_neg:(train_size_neg+val_size_neg)], neg_k[train_size_neg:(train_size_neg+val_size_neg)], neg_l[train_size_neg:(train_size_neg+val_size_neg)]
        
        test_neg_i, test_neg_j, test_neg_k, test_neg_l = neg_i[(train_size_neg+val_size_neg):], neg_j[(train_size_neg+val_size_neg):], neg_k[(train_size_neg+val_size_neg):], neg_l[(train_size_neg+val_size_neg):] 
        
        print(' #triangles of training positive tetras:', len(train_pos_i), len(train_pos_j), len(train_pos_k), len(train_pos_l), '\n','#triangles of training negative tetras:', len(train_neg_i), len(train_neg_j), len(train_neg_k), len(train_neg_l), '\n',
        '#triangles of testing positive tetras:', len(test_pos_i), len(test_pos_j), len(test_pos_k), len(test_pos_l), '\n','#triangles of testing negative tetras:', len(test_neg_i), len(test_neg_j), len(test_neg_k), len(test_neg_l))

        '''
            now we perform the triangle prediction on the test dataset with HARMONIC MEAN, ARITHMETIC and GEOMETRIC means method
        '''
        f_tri = np.array(cochains[2])
    
        harm_pos_score = 1/((1/f_tri[test_pos_i] + 1/f_tri[test_pos_j] + 1/f_tri[test_pos_k] + 1/f_tri[test_pos_l])/4)
        arith_pos_score = (f_tri[test_pos_i] + f_tri[test_pos_j] + f_tri[test_pos_k] + f_tri[test_pos_l])/4

        x = sp.Symbol('x')
        geom_pos_score = [sp.limit(((f_tri[test_pos_i][i]**x + f_tri[test_pos_j][i]**x + f_tri[test_pos_k][i]**x + f_tri[test_pos_l][i]**x)/4)**(1/x),x,0) for i in range(len(test_pos_i))]
        
        
        harm_neg_score = 1/((1/f_tri[test_neg_i] + 1/f_tri[test_neg_j] + 1/f_tri[test_neg_k] + 1/f_tri[test_neg_l])/4)
        arith_neg_score = (f_tri[test_neg_i] + f_tri[test_neg_j] + f_tri[test_neg_k]+ f_tri[test_neg_l])/4

        x = sp.Symbol('x')
        geom_neg_score = [sp.limit(((f_tri[test_neg_i][i]**x + f_tri[test_neg_j][i]**x + f_tri[test_neg_k][i]**x + f_tri[test_neg_l][i]**x)/4)**(1/x),x,0) for i in range(len(test_neg_i))]

        geom_pos_score = [float(geom_pos_score[i]) for i in range(len(geom_pos_score))]
        geom_neg_score = [float(geom_neg_score[i]) for i in range(len(geom_neg_score))]
        
        # print(geom_pos_score)
        # print(geom_neg_score)
        
        harm_auc.append(compute_auc(torch.tensor(harm_pos_score),torch.tensor(harm_neg_score)))
        arith_auc.append(compute_auc(torch.tensor(arith_pos_score),torch.tensor(arith_neg_score)))
        geom_auc.append(compute_auc(torch.tensor(geom_pos_score),torch.tensor(geom_neg_score)))

    print('harmomic mean AUC: ', harm_auc, '\n' 'arithmetic mean AUC: ', arith_auc, '\n' 'geometric mean AUC: ', geom_auc)
    print('harm.: ', np.mean(harm_auc), np.std(harm_auc))
    print('arit.: ', np.mean(arith_auc), np.std(arith_auc))
    print('geom.: ', np.mean(geom_auc), np.std(geom_auc))
    
if __name__ == "__main__":
    main()