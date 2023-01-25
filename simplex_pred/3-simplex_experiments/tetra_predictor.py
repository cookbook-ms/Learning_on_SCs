
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score 


def compute_loss(pos_score, neg_score,labels):
    # the edge scores computed, concatenate them 
    scores = torch.cat([pos_score, neg_score])
    return F.binary_cross_entropy_with_logits(scores,labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels,scores)


class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        # the first layer of MLP is the concatenated version of the three edge features which have dimension of h_feats
        self.W1 = nn.Linear(h_feats * 4, h_feats) 
        self.W2 = nn.Linear(h_feats, 1)
        self.m1 = nn.LeakyReLU(0.1)
        self.m2 = nn.Sigmoid()

    def forward(self, fi, fj, fk, fl):
        """
        Computes a scalar score for each triangle 

        Parameters
        ----------
        edge features
        fi -- features of the first triangle, dim: #training/testing pos/neg tetras x # features 
        fj -- features of the second triangle, dim: #training/testing pos/neg tetras x # features  
        fk -- features of the third triangle, dim: #training/testing pos/neg tetras x # features  
        fl -- features of the fourth triangle, dim: #training/testing pos/neg tetras x # features  

        Returns
        -------
        new feature or score used to perform prediction
        """
        h = torch.cat([fi, fj, fk, fl], 1)
        return (self.W2(self.m2(self.W1(h)))).squeeze(1)


class MLPPredictor_edge(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        # the first layer of MLP is the concatenated version of the three edge features which have dimension of h_feats
        self.W1 = nn.Linear(h_feats * 6, h_feats) 
        self.W2 = nn.Linear(h_feats, 1)
        self.m1 = nn.LeakyReLU(0.1)
        self.m2 = nn.Sigmoid()

    def forward(self, f1,f2,f3,f4,f5,f6):
        """
        Computes a scalar score for each triangle 

        Parameters
        ----------
        edge features
        fi -- features of the first triangle, dim: #training/testing pos/neg tetras x # features 
        fj -- features of the second triangle, dim: #training/testing pos/neg tetras x # features  
        fk -- features of the third triangle, dim: #training/testing pos/neg tetras x # features  
        fl -- features of the fourth triangle, dim: #training/testing pos/neg tetras x # features  

        Returns
        -------
        new feature or score used to perform prediction
        """
        h = torch.cat([f1,f2,f3,f4,f5,f6], 1)
        return (self.W2(self.m2(self.W1(h)))).squeeze(1)
