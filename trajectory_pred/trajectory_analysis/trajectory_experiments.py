'''
This is the code to perform prediction using different methods 

built based on the paper: https://arxiv.org/abs/2102.10058v3
'''

###########################################
"""
This is where you actually train models. See below for docs:

Generate a synthetic graph, with holes in upper left and lower right regions, + paths over the graph:
    python3 synthetic_data_gen.py
    -Edit main of synthetic_data_gen.py to change graph size / # of paths

Train a SCoNe model on a dataset:
    python3 trajectory_experiments.py [args]

    -Command to run standard training / experiment with defaults:
        python3 trajectory_experiments.py -data_folder_suffix suffix_here

    -The default hyperparameters should work pretty well on the default graph size. You'll probably have to play with
        them for other graphs, though.

Arguments + default values for trajectory_experiments.py:
   'model': 'scone'; which model to use, of ('scone', 'ebli', or 'bunch')
        -'scone': ours
        -'ebli':  https://arxiv.org/pdf/2010.03633.pdf
        -'bunch': https://arxiv.org/pdf/2012.06010.pdf
   'epochs': 1000; # of training epochs
   'learning_rate': 0.001; starting learning rate
   'weight_decay': 0.00005; ridge regularization constant
   'batch_size': 100; # of samples per batch (randomly selected)
   'reverse': 0;  if 1, also compute accuracy over the test set, but reversed (Reverse experiment)
   'data_folder_suffix': 'schaub2'; set suffix of folder to import data from (trajectory_data_Nhop_suffix)
   'regional': 0; if 1, trains a model over upper graph region and tests over lower region (Transfer experiment)

   'hidden_layers': 3_16_3_16_3_16 (corresponds to [(3, 16), (3, 16), (3, 16)]; each tuple is a layer (# of shift matrices, # of units in layer) )
        -'scone' and 'ebli' require 3_#_3_#_ ...; 
        -'scnn' require 3_#_3_#_ ..., its automated by k1 and k2
        -'bunch' requires 7_#_7_#_ ...
        -'sccnn1' requires 15_#_15_#_...
        -'sccnn2' requires 23_#_23_#_...
   'describe': 1; describes the dataset being used
   'load_data': 1; if 0, generate new data; if 1, load data from folder set in data_folder_suffix
   'load_model': 0; if 0, train a new model, if 1, load model from file model_name.npy. Must set hidden_layers regardless of choice
   'markov': 0; include tests using a 2nd-order Markov model
   'model_name': 'model'; name of model to use when load_model = 1
   'flip_edges': 0; if 1, flips orientation of a random subset of edges. with tanh activation, should perform equally
   'multi_graph': '': if not '', also tests on paths over the graph with the folder suffix set here
   'holes': 1; if generation new data, sets whether the graph should have holes

More examples:
    python3 trajectory_experiments.py -model_name tanh -reverse 1 -epochs 1100 -load_model 1 -multi_graph no_holes
        -loads model tanh.npy from models folder, tests it over reversed test set, and also tests over another graph saved in trajectory_data_Nhop_no_holes
    python3 trajectory_experiments.py load_data 0 -holes 0 -model_name tanh_no_holes -hidden_layers [(3, 32), (3,16)] -data_folder_suffix no_holes2
        -generates a new graph with no holes; saves dataset to trajectory_data_Nhop_no_holes2;
            trains a new model with 2 layers (32 and 16 channels, respectively) for 1100 epochs, and saves its weights to tanh_no_holes.npy
    python3 trajectory_experiments.py -load_data 0 -holes 1 -data_folder_suffix holes
        -make a dataset with holes, save with folder suffix holes (just stop the program once training starts if you just want to make a new dataset)
    python3 trajectory_experiments.py load_data 0 -holes 0 -data_folder_suffix no_holes -model_name tanh_no_holes -multi_graph holes
        -create a dataset using folder suffix no_holes, train a model over it using default settings, and test it over the graph with data folder suffix holes
"""
from cmath import tan
import os, sys
import numpy as onp
from numpy import linalg as la
import jax.numpy as np
from jax.scipy.special import logsumexp
from pathlib import Path
path = Path(__file__).parent.absolute()
os.chdir(path)

from bunch_model_matrices import compute_shift_matrices, compute_shift_matrices_sccnn1
from synthetic_data_gen import load_dataset, generate_dataset, neighborhood, conditional_incidence_matrix, flow_to_path
from scone_trajectory_model import Scone_GCN


def hyperparams():
    """
    Parse hyperparameters from command line

    For hidden_layers, input [(3, 8), (3, 8)] as 3_8_3_8
    """
    args = sys.argv
    hyperparams = {'model': 'scone',
                   'epochs': 1000,
                   'learning_rate': 0.001,
                   'weight_decay': 0.000005,
                   'batch_size': 100,
                   'hidden_layers': [(3, 16), (3, 16), (3, 16)], # where 3 indicates the 1 identity, 1 lower and 1 upper shift # for ebli, replace 3 by 4, for bunch, replace 3 by 7, for scnn, no need to replace, for scnn1, replace 3 by 15, for sccnn2, replace 3 by 23
                   'k1_scnn': 2,
                   'k2_scnn': 2,
                   'describe': 1,
                   'reverse': 1,
                   'load_data': 1,
                   'load_model': 0,
                   'normalized': 1, # if to normalize the hodge laplacians
                   'markov': 0,
                   'model_name': 'model',
                   'regional': 0,
                   'flip_edges': 0,
                   'data_folder_suffix': 'working2_5',
                   'multi_graph': '',
                   'holes': 1}

    for i in range(len(args) - 1):
        if args[i][0] == '-':
            if args[i][1:] == 'hidden_layers':
                nums = list(map(int, args[i + 1].split("_")))
                hyperparams['hidden_layers'] = []
                for j in range(0, len(nums), 2):
                    hyperparams['hidden_layers'] += [(nums[j], nums[j + 1])]
            elif args[i][1:] in ['model_name', 'data_folder_suffix', 'multi_graph', 'model']:
                hyperparams[args[i][1:]] = str(args[i+1])
            elif args[i][1:] in ['k1_scnn','k2_scnn']:
                hyperparams[args[i][1:]] = int(args[i+1])
            else:
                hyperparams[args[i][1:]] = float(args[i+1])


    return hyperparams

HYPERPARAMS = hyperparams()

### Model definition ###

# Activation functions
def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def leaky_relu(x):
    return np.where(x >= 0, x, 0.01 * x)

# SCoNe function
def scone_func(weights, S_lower, S_upper, Bcond_func, last_node, flow):
    """
    Forward pass of the SCoNe model with variable number of layers
    """
    n_layers = (len(weights) - 1) / 3
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = flow
    for i in range(int(n_layers)):
        cur_out = cur_out @ weights[i * 3] \
                  + S_lower @ cur_out @ weights[i*3 + 1] \
                  + S_upper @ cur_out @ weights[i*3 + 2]

        cur_out = tanh(cur_out)

    logits = Bcond_func(last_node) @ cur_out @ weights[-1]
    return logits - logsumexp(logits) # log of the softmax function 


def scnn_func_2(weights, S_lower, S2_lower, S_upper, S2_upper, Bcond_func, last_node, flow, k1=HYPERPARAMS['k1_scnn'], k2=HYPERPARAMS['k2_scnn']):
    """
    Forward pass of the SCoNe model with variable number of layers
    """
    n_layers = (len(weights) - 1) / (1+k1+k2) # k1=k2=2
    n_k = 1+k1+k2
    #print('#weights:',len(weights),'#layers:',n_layers)
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = flow
    for i in range(int(n_layers)):
        cur_out = cur_out @ weights[i*n_k] \
                  + S_lower @ cur_out @ weights[i*n_k + 1] \
                  + S2_lower @ cur_out @ weights[i*n_k + 2] \
                  + S_upper @ cur_out @ weights[i*n_k + 3] \
                  + S2_upper @ cur_out @ weights[i*n_k +4]    

        cur_out = tanh(cur_out)

    logits = Bcond_func(last_node) @ cur_out @ weights[-1]
    return logits - logsumexp(logits)

def scnn_func_12(weights, S_lower, S_upper, S2_upper, Bcond_func, last_node, flow, k1=HYPERPARAMS['k1_scnn'], k2=HYPERPARAMS['k2_scnn']):
    """
    Forward pass of the SCoNe model with variable number of layers
    """
    n_layers = (len(weights) - 1) / (1+k1+k2) # k1=k2=2
    n_k = 1+k1+k2
    #print('#weights:',len(weights),'#layers:',n_layers)
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = flow
    for i in range(int(n_layers)):
        cur_out = cur_out @ weights[i*n_k] \
                  + S_lower @ cur_out @ weights[i*n_k + 1] \
                  + S_upper @ cur_out @ weights[i*n_k + 2] \
                  + S2_upper @ cur_out @ weights[i*n_k + 3]   

        cur_out = tanh(cur_out)

    logits = Bcond_func(last_node) @ cur_out @ weights[-1]
    return logits - logsumexp(logits)


def scnn_func_21(weights, S_lower, S2_lower, S_upper, Bcond_func, last_node, flow, k1=HYPERPARAMS['k1_scnn'], k2=HYPERPARAMS['k2_scnn']):
    """
    Forward pass of the SCoNe model with variable number of layers
    """
    n_layers = (len(weights) - 1) / (1+k1+k2) # k1=k2=2
    n_k = 1+k1+k2
    #print('#weights:',len(weights),'#layers:',n_layers)
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = flow
    for i in range(int(n_layers)):
        cur_out = cur_out @ weights[i*n_k] \
                  + S_lower @ cur_out @ weights[i*n_k + 1] \
                  + S2_lower @ cur_out @ weights[i*n_k + 2] \
                  + S_upper @ cur_out @ weights[i*n_k + 3]   

        cur_out = tanh(cur_out)

    logits = Bcond_func(last_node) @ cur_out @ weights[-1]
    return logits - logsumexp(logits)

def scnn_func_13(weights, S_lower, S_upper, S2_upper, S3_upper, Bcond_func, last_node, flow, k1=HYPERPARAMS['k1_scnn'], k2=HYPERPARAMS['k2_scnn']):
    """
    Forward pass of the SCoNe model with variable number of layers
    """
    n_layers = (len(weights) - 1) / (1+k1+k2) # k1=k2=2
    n_k = 1+k1+k2
    #print('#weights:',len(weights),'#layers:',n_layers)
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = flow
    for i in range(int(n_layers)):
        cur_out = cur_out @ weights[i*n_k] \
                  + S_lower @ cur_out @ weights[i*n_k + 1] \
                  + S_upper @ cur_out @ weights[i*n_k + 2] \
                  + S2_upper @ cur_out @ weights[i*n_k + 3] \
                  + S3_upper @ cur_out @ weights[i*n_k +4]    

        cur_out = tanh(cur_out)

    logits = Bcond_func(last_node) @ cur_out @ weights[-1]
    return logits - logsumexp(logits)


def scnn_func_31(weights, S_lower, S2_lower, S3_lower, S_upper, Bcond_func, last_node, flow, k1=HYPERPARAMS['k1_scnn'], k2=HYPERPARAMS['k2_scnn']):
    """
    Forward pass of the SCoNe model with variable number of layers
    """
    n_layers = (len(weights) - 1) / (1+k1+k2) # k1=k2=2
    n_k = 1+k1+k2
    #print('#weights:',len(weights),'#layers:',n_layers)
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = flow
    for i in range(int(n_layers)):
        cur_out = cur_out @ weights[i*n_k] \
                  + S_lower @ cur_out @ weights[i*n_k + 1] \
                  + S2_lower @ cur_out @ weights[i*n_k + 2] \
                  + S3_lower @ cur_out @ weights[i*n_k + 3] \
                  + S_upper @ cur_out @ weights[i*n_k +4]    

        cur_out = tanh(cur_out)

    logits = Bcond_func(last_node) @ cur_out @ weights[-1]
    return logits - logsumexp(logits)


def scnn_func_3(weights, S_lower, S2_lower, S3_lower, S_upper, S2_upper, S3_upper, Bcond_func, last_node, flow, k1=HYPERPARAMS['k1_scnn'], k2=HYPERPARAMS['k2_scnn']):
    """
    Forward pass of the SCoNe model with variable number of layers
    """
    n_layers = (len(weights) - 1) / (1+k1+k2) # k1=k2=3
    n_k = 1+k1+k2
    #print('#weights:',len(weights),'#layers:',n_layers)
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = flow
    for i in range(int(n_layers)):
        cur_out = cur_out @ weights[i*n_k] \
                  + S_lower @ cur_out @ weights[i*n_k + 1] \
                  + S2_lower @ cur_out @ weights[i*n_k + 2] \
                  + S3_lower @ cur_out @ weights[i*n_k + 3] \
                  + S_upper @ cur_out @ weights[i*n_k + 4] \
                  + S2_upper @ cur_out @ weights[i*n_k +5] \
                  + S3_upper @ cur_out @ weights[i*n_k + 6]       

        cur_out = tanh(cur_out)

    logits = Bcond_func(last_node) @ cur_out @ weights[-1]
    return logits - logsumexp(logits)

def scnn_func_23(weights, S_lower, S2_lower, S_upper, S2_upper, S3_upper, Bcond_func, last_node, flow, k1=HYPERPARAMS['k1_scnn'], k2=HYPERPARAMS['k2_scnn']):
    """
    Forward pass of the SCoNe model with variable number of layers
    """
    n_layers = (len(weights) - 1) / (1+k1+k2) # k1=k2=3
    n_k = 1+k1+k2
    #print('#weights:',len(weights),'#layers:',n_layers)
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = flow
    for i in range(int(n_layers)):
        cur_out = cur_out @ weights[i*n_k] \
                  + S_lower @ cur_out @ weights[i*n_k + 1] \
                  + S2_lower @ cur_out @ weights[i*n_k + 2] \
                  + S_upper @ cur_out @ weights[i*n_k + 3] \
                  + S2_upper @ cur_out @ weights[i*n_k + 4] \
                  + S3_upper @ cur_out @ weights[i*n_k + 5]       

        cur_out = tanh(cur_out)

    logits = Bcond_func(last_node) @ cur_out @ weights[-1]
    return logits - logsumexp(logits)


def scnn_func_32(weights, S_lower, S2_lower, S3_lower, S_upper, S2_upper, Bcond_func, last_node, flow, k1=HYPERPARAMS['k1_scnn'], k2=HYPERPARAMS['k2_scnn']):
    """
    Forward pass of the SCoNe model with variable number of layers
    """
    n_layers = (len(weights) - 1) / (1+k1+k2) # k1=k2=3
    n_k = 1+k1+k2
    #print('#weights:',len(weights),'#layers:',n_layers)
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = flow
    for i in range(int(n_layers)):
        cur_out = cur_out @ weights[i*n_k] \
                  + S_lower @ cur_out @ weights[i*n_k + 1] \
                  + S2_lower @ cur_out @ weights[i*n_k + 2] \
                  + S3_lower @ cur_out @ weights[i*n_k + 3] \
                  + S_upper @ cur_out @ weights[i*n_k + 4] \
                  + S2_upper @ cur_out @ weights[i*n_k +5]     

        cur_out = tanh(cur_out)

    logits = Bcond_func(last_node) @ cur_out @ weights[-1]
    return logits - logsumexp(logits)

def scnn_func_4(weights, S_lower, S2_lower, S3_lower, S4_lower, S_upper, S2_upper, S3_upper, S4_upper, Bcond_func, last_node, flow, k1=HYPERPARAMS['k1_scnn'], k2=HYPERPARAMS['k2_scnn']):
    """
    Forward pass of the SCoNe model with variable number of layers
    """
    n_layers = (len(weights) - 1) / (1+k1+k2) # k1=k2=4
    n_k = 1+k1+k2
    #print('#weights:',len(weights),'#layers:',n_layers)
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = flow
    for i in range(int(n_layers)):
        cur_out = cur_out @ weights[i*n_k] \
                  + S_lower @ cur_out @ weights[i*n_k + 1] \
                  + S2_lower @ cur_out @ weights[i*n_k + 2] \
                  + S3_lower @ cur_out @ weights[i*n_k + 3] \
                  + S4_lower @ cur_out @ weights[i*n_k + 4] \
                  + S_upper @ cur_out @ weights[i*n_k + 5] \
                  + S2_upper @ cur_out @ weights[i*n_k + 6] \
                  + S3_upper @ cur_out @ weights[i*n_k + 7] \
                  + S4_upper @ cur_out @ weights[i*n_k + 8]         

        cur_out = tanh(cur_out)

    logits = Bcond_func(last_node) @ cur_out @ weights[-1]
    return logits - logsumexp(logits)


def scnn_func_5(weights, S_lower, S2_lower, S3_lower, S4_lower, S5_lower, S_upper, S2_upper, S3_upper, S4_upper, S5_upper, Bcond_func, last_node, flow, k1=HYPERPARAMS['k1_scnn'], k2=HYPERPARAMS['k2_scnn']):
    """
    Forward pass of the SCoNe model with variable number of layers
    """
    n_layers = (len(weights) - 1) / (1+k1+k2) # k1=k2=4
    n_k = 1+k1+k2
    #print('#weights:',len(weights),'#layers:',n_layers)
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = flow
    for i in range(int(n_layers)):
        cur_out = cur_out @ weights[i*n_k] \
                  + S_lower @ cur_out @ weights[i*n_k + 1] \
                  + S2_lower @ cur_out @ weights[i*n_k + 2] \
                  + S3_lower @ cur_out @ weights[i*n_k + 3] \
                  + S4_lower @ cur_out @ weights[i*n_k + 4] \
                  + S5_lower @ cur_out @ weights[i*n_k + 5] \
                  + S_upper @ cur_out @ weights[i*n_k + 6] \
                  + S2_upper @ cur_out @ weights[i*n_k + 7] \
                  + S3_upper @ cur_out @ weights[i*n_k + 8] \
                  + S4_upper @ cur_out @ weights[i*n_k + 9] \
                  + S5_upper @ cur_out @ weights[i*n_k + 10]        

        cur_out = tanh(cur_out)

    logits = Bcond_func(last_node) @ cur_out @ weights[-1]
    return logits - logsumexp(logits)


# Ebli function
def ebli_func(weights, S, S2, S3, Bcond_func, last_node, flow):
    """
    Forward pass of the Ebli model with variable number of layers
    note that here 
    S_lower = L1
    S_upper = L1^2
    """
    n_layers = (len(weights) - 1) / 4
    #print('#weights:',len(weights),'#layers:',n_layers)
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = flow
    for i in range(int(n_layers)):
        cur_out = cur_out @ weights[i * 4] \
                  + S @ cur_out @ weights[i*4 + 1] \
                  + S2 @ cur_out @ weights[i*4 + 2] \
                  + S3 @ cur_out @ weights[i*4 + 3]

        cur_out = tanh(cur_out)

    logits = Bcond_func(last_node) @ cur_out @ weights[-1]
    return logits - logsumexp(logits)

# Bunch function
def bunch_func(weights, S_00, S_10, S_01, S_11, S_21, S_12, S_22, nbrhoods, last_node, flow):
    """
    Forward pass of the Bunch model with variable number of layers
    """
    n_layers = (len(weights)) / 7
    #print('#weights:',len(weights),'#layers:',n_layers)
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = [np.zeros((S_00.shape[1], 1)), flow, np.zeros((S_22.shape[1], 1))]

    for i in range(int(n_layers)):
        next_out = [None, None, None]
        # node level
        next_out[0] = S_00 @ cur_out[0] @ weights[i * 7] \
                   + S_10 @ cur_out[1] @ weights[i * 7 + 1]

        next_out[1] = S_01 @ cur_out[0] @ weights[i * 7 + 2] \
                   + S_11 @ cur_out[1] @ weights[i * 7 + 3] \
                   + S_21 @ cur_out[2] @ weights[i * 7 + 4]

        next_out[2] = S_12 @ cur_out[1] @ weights[i * 7 + 5] \
                   + S_22 @ cur_out[2] @ weights[i * 7 + 6]

        cur_out = [tanh(c) for c in next_out]
    '''option 1'''
    nodes_out = cur_out[0] # use the last layer output on the node level as the final output 
    # values at nbrs of last node
    logits = nodes_out[nbrhoods[last_node]]
    return logits - logsumexp(logits)


def bunch_func_2(weights, S_00, S_10, S_01, S_11, S_21, S_12, S_22, Bcond_func, last_node, flow):
    """
    Forward pass of the Bunch model with variable number of layers
    """
    n_layers = (len(weights)-1) / 7
    #print('#weights:',len(weights),'#layers:',n_layers)
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = [np.zeros((S_00.shape[1], 1)), flow, np.zeros((S_22.shape[1], 1))]

    for i in range(int(n_layers)):
        next_out = [None, None, None]
        # node level
        next_out[0] = S_00 @ cur_out[0] @ weights[i * 7] \
                   + S_10 @ cur_out[1] @ weights[i * 7 + 1]

        next_out[1] = S_01 @ cur_out[0] @ weights[i * 7 + 2] \
                   + S_11 @ cur_out[1] @ weights[i * 7 + 3] \
                   + S_21 @ cur_out[2] @ weights[i * 7 + 4]

        next_out[2] = S_12 @ cur_out[1] @ weights[i * 7 + 5] \
                   + S_22 @ cur_out[2] @ weights[i * 7 + 6]

        cur_out = [tanh(c) for c in next_out]
    '''option 2'''
    # let us consider the edge features then project them to the node space like other models
    logits = Bcond_func(last_node) @ cur_out[1] @ weights[-1]
    return logits - logsumexp(logits)

# sccnn1 function
def sccnn_func_1_n(weights, S_00,S_10,  S_01, S_11_d, S_11_u,S_21,  S_12,S_22, nbrhoods, last_node, flow):
    """
    Forward pass of the sccnn1 model with variable number of layers: here we consider the laplacians as shift operators for consisten comparisons with scnn and snn and psnn
    """
    ns = 15 # number of shift operators in total including identity matrices
    n_layers = (len(weights)) / ns
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = [np.zeros((S_00.shape[1], 1)), flow, np.zeros((S_22.shape[1], 1))]

    for i in range(int(n_layers)):
        next_out = [None, None, None]
        # node level
        next_out[0] = cur_out[0] @ weights[i*ns] + S_00 @ cur_out[0] @ weights[i*ns+1] \
                    + S_10 @ cur_out[1] @ weights[i*ns+2] + S_00 @ S_10 @ cur_out[1] @ weights[i*ns+3] 

        next_out[1] = S_01 @ cur_out[0] @ weights[i*ns+4] + S_11_d @ S_01 @ cur_out[0] @ weights[i*ns+5] \
                   + cur_out[1] @ weights[i*ns+6] + S_11_d @ cur_out[1] @ weights[i*ns+7] + S_11_u @ cur_out[1] @ weights[i*ns+8] \
                   + S_21 @ cur_out[2] @ weights[i*ns+9] + S_11_u @ S_21 @ cur_out[2] @ weights[i*ns+10] 

        next_out[2] = S_12 @ cur_out[1] @ weights[i*ns+11] + S_22 @ S_12 @ cur_out[1] @ weights[i*ns+12]\
                   + cur_out[2] @ weights[i*ns+13] + S_22 @ cur_out[2] @ weights[i*ns+14]

        cur_out = [tanh(c) for c in next_out]
    '''option 1'''
    nodes_out = cur_out[0] # use the last layer output on the node level as the final output 
    # values at nbrs of last node
    logits = nodes_out[nbrhoods[last_node]]
    return logits - logsumexp(logits)

# sccnn2 function
def sccnn_func_2_n(weights, S_00,S_10,  S_01, S_11_d, S_11_u,S_21,  S_12,S_22, nbrhoods, last_node, flow):
    """
    Forward pass of the sccnn2 model with variable number of layers: here we consider the laplacians as shift operators for consisten comparisons with scnn and snn and psnn
    """
    ns = 23 # number of shift operators in total including identity matrices
    n_layers = (len(weights)) / ns
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = [np.zeros((S_00.shape[1], 1)), flow, np.zeros((S_22.shape[1], 1))]

    for i in range(int(n_layers)):
        next_out = [None, None, None]
        # node level
        next_out[0] = cur_out[0] @ weights[i*ns] + S_00 @ cur_out[0] @ weights[i*ns+1] + S_00 @ S_00 @ cur_out[0] @ weights[i*ns+2] \
                    + S_10 @ cur_out[1] @ weights[i*ns+3] + S_00 @ S_10 @ cur_out[1] @ weights[i*ns+4] + S_00 @ S_00 @ S_10 @ cur_out[1] @ weights[i*ns+5] 

        next_out[1] = S_01 @ cur_out[0] @ weights[i*ns+6] + S_11_d @ S_01 @ cur_out[0] @ weights[i*ns+7] + S_11_d @ S_11_d @ S_01 @ cur_out[0] @ weights[i*ns+8] \
                   + cur_out[1] @ weights[i*ns+9] + S_11_d @ cur_out[1] @ weights[i*ns+10] + S_11_d @ S_11_d @ cur_out[1] @ weights[i*ns+11] + S_11_u @ cur_out[1] @ weights[i*ns+12] + S_11_u @ S_11_u @ cur_out[1] @ weights[i*ns+13] \
                   + S_21 @ cur_out[2] @ weights[i*ns+14] + S_11_u @ S_21 @ cur_out[2] @ weights[i*ns+15] + S_11_u @ S_11_u @ S_21 @ cur_out[2] @ weights[i*ns+16]

        next_out[2] = S_12 @ cur_out[1] @ weights[i*ns+17] + S_22 @ S_12 @ cur_out[1] @ weights[i*ns+18] + S_22 @ S_22 @ S_12 @ cur_out[1] @ weights[i*ns+19]\
                   + cur_out[2] @ weights[i*ns+20] + S_22 @ cur_out[2] @ weights[i*ns+21] + S_22 @ S_22 @ cur_out[2] @ weights[i*ns+22]

        cur_out = [tanh(c) for c in next_out]
    '''option 1'''
    nodes_out = cur_out[0] # use the last layer output on the node level as the final output 
    # values at nbrs of last node
    logits = nodes_out[nbrhoods[last_node]]
    return logits - logsumexp(logits)

def sccnn_func_1_e(weights, S_00,S_10,  S_01, S_11_d, S_11_u, S_21,  S_12,S_22, Bcond_func, last_node, flow):
    """
    Forward pass of the sccnn1 model with variable number of layers: here we consider the laplacians as shift operators for consisten comparisons with scnn and snn and psnn
    """
    ns = 15 # number of shift operators in total including identity matrices
    n_layers = (len(weights)-1) / ns
    #print('#weights:',len(weights),'#layers:',n_layers)
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = [np.zeros((S_00.shape[1], 1)), flow, np.zeros((S_22.shape[1], 1))]

    for i in range(int(n_layers)):
        next_out = [None, None, None]
        # node level
        next_out[0] = cur_out[0] @ weights[i*ns] + S_00 @ cur_out[0] @ weights[i*ns+1] \
                    + S_10 @ cur_out[1] @ weights[i*ns+2] + S_00 @ S_10 @ cur_out[1] @ weights[i*ns+3] 

        next_out[1] = S_01 @ cur_out[0] @ weights[i*ns+4] + S_11_d @ S_01 @ cur_out[0] @ weights[i*ns+5] \
                   + cur_out[1] @ weights[i*ns+6] + S_11_d @ cur_out[1] @ weights[i*ns+7] + S_11_u @ cur_out[1] @ weights[i*ns+8] \
                   + S_21 @ cur_out[2] @ weights[i*ns+9] + S_11_u @ S_21 @ cur_out[2] @ weights[i*ns+10]

        next_out[2] = S_12 @ cur_out[1] @ weights[i*ns+11] + S_22 @ S_12 @ cur_out[1] @ weights[i*ns+12]\
                   + cur_out[2] @ weights[i*ns+13] + S_22 @ cur_out[2] @ weights[i*ns+14]

        cur_out = [tanh(c) for c in next_out]
    '''option 2'''
    # let us consider the edge features then project them to the node space like other models
    logits = Bcond_func(last_node) @ cur_out[1] @ weights[-1]
    return logits - logsumexp(logits)


def sccnn_func_2_e(weights, S_00,S_10,  S_01, S_11_d, S_11_u, S_21,  S_12,S_22, Bcond_func, last_node, flow):
    """
    Forward pass of the sccnn1 model with variable number of layers: here we consider the laplacians as shift operators for consisten comparisons with scnn and snn and psnn
    """
    ns = 23 # number of shift operators in total including identity matrices
    n_layers = (len(weights)-1) / ns
    #print('#weights:',len(weights),'#layers:',n_layers)
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = [np.zeros((S_00.shape[1], 1)), flow, np.zeros((S_22.shape[1], 1))]

    for i in range(int(n_layers)):
        next_out = [None, None, None]
        # node level
        next_out[0] = cur_out[0] @ weights[i*ns] + S_00 @ cur_out[0] @ weights[i*ns+1] + S_00 @ S_00 @ cur_out[0] @ weights[i*ns+2] \
                    + S_10 @ cur_out[1] @ weights[i*ns+3] + S_00 @ S_10 @ cur_out[1] @ weights[i*ns+4] + S_00 @ S_00 @ S_10 @ cur_out[1] @ weights[i*ns+5] \

        next_out[1] = S_01 @ cur_out[0] @ weights[i*ns+6] + S_11_d @ S_01 @ cur_out[0] @ weights[i*ns+7] + S_11_d @ S_11_d @ S_01 @ cur_out[0] @ weights[i*ns+8] \
                   + cur_out[1] @ weights[i*ns+9] + S_11_d @ cur_out[1] @ weights[i*ns+10] + S_11_d @ S_11_d @ cur_out[1] @ weights[i*ns+11] + S_11_u @ cur_out[1] @ weights[i*ns+12] + S_11_u @ S_11_u @ cur_out[1] @ weights[i*ns+13] \
                   + S_21 @ cur_out[2] @ weights[i*ns+14] + S_11_u @ S_21 @ cur_out[2] @ weights[i*ns+15] + S_11_u @ S_11_u @ S_21 @ cur_out[2] @ weights[i*ns+16]

        next_out[2] = S_12 @ cur_out[1] @ weights[i*ns+17] + S_22 @ S_12 @ cur_out[1] @ weights[i*ns+18] + S_22 @ S_22 @ S_12 @ cur_out[1] @ weights[i*ns+19]\
                   + cur_out[2] @ weights[i*ns+20] + S_22 @ cur_out[2] @ weights[i*ns+21] + S_22 @ S_22 @ cur_out[2] @ weights[i*ns+22]

        cur_out = [tanh(c) for c in next_out]
    '''option 2'''
    # let us consider the edge features then project them to the node space like other models
    logits = Bcond_func(last_node) @ cur_out[1] @ weights[-1]
    return logits - logsumexp(logits)





def data_setup(hops=(1,), load=True, folder_suffix='working0'):
    """
    Imports and sets up flow, target, and shift matrices for model training. Supports generating data for multiple hops
        at once
    """

    inputs_all, y_all, target_nodes_all = [], [], []

    if HYPERPARAMS['flip_edges']:
        # Flip orientation of a random subset of edges
        onp.random.seed(1)
        _, _, _, _, _, G_undir, _, _ = load_dataset('../data/trajectory_data_1hop_' + folder_suffix)
        flips = onp.random.choice([1, -1], size=len(G_undir.edges), replace=True, p=[0.8, 0.2])
        F = np.diag(flips)


    if not load:
        # Generate new data
        generate_dataset(400, 1000, folder=folder_suffix, holes=HYPERPARAMS['holes'])
        raise Exception('Data generation done')


    for h in hops:
        # Load data
        folder = '../data/trajectory_data_' + str(h) + 'hop_' + folder_suffix
        X, B_matrices, y, train_mask, test_mask, G_undir, last_nodes, target_nodes = load_dataset(folder)
        B1, B2 = B_matrices
        # compute the diagonal matrices in the hodge laplacian normalization 
        D2 = np.maximum(np.diag(abs(B2)@np.ones(B2.shape[1])),np.identity(B2.shape[0]))
        D1 = 2 * np.diag(abs(B1)@D2@np.ones(D2.shape[0]))
        D3 = 1/3*np.identity(B2.shape[1])
        target_nodes_all.append(target_nodes)
        
        inputs_all.append([None, onp.array(last_nodes), X])
        y_all.append(y)

        # Define shifts
        if HYPERPARAMS['normalized']:
            L1_lower = D2 @ B1.T @ la.pinv(D1) @ B1 
            L1_upper = B2 @ D3 @ B2.T @ la.inv(D2)
        else: 
            L1_lower = B1.T @ B1
            L1_upper = B2 @ B2.T
        
        if HYPERPARAMS['flip_edges']:
            L1_lower = F @ L1_lower @ F
            L1_upper = F @ L1_upper @ F

        if HYPERPARAMS['model'] == 'scone':
            shifts = [L1_lower, L1_upper]
        elif HYPERPARAMS['model'] == 'scnn2':
            shifts = [L1_lower, L1_lower@L1_lower, L1_upper, L1_upper@L1_upper]
        elif HYPERPARAMS['model'] == 'scnn12':
            shifts = [L1_lower, L1_upper, L1_upper@L1_upper]
        elif HYPERPARAMS['model'] == 'scnn21':
            shifts = [L1_lower, L1_lower@L1_lower, L1_upper]
        elif HYPERPARAMS['model'] == 'scnn13':
            shifts = [L1_lower, L1_upper, L1_upper@L1_upper, L1_upper@L1_upper@L1_upper]
        elif HYPERPARAMS['model'] == 'scnn31':
            shifts = [L1_lower, L1_lower@L1_lower, L1_lower@L1_lower@L1_lower, L1_upper]
        elif HYPERPARAMS['model'] == 'scnn23':
            shifts = [L1_lower, L1_lower@L1_lower, L1_upper, L1_upper@L1_upper, L1_upper@L1_upper@L1_upper]  
        elif HYPERPARAMS['model'] == 'scnn32':
            shifts = [L1_lower, L1_lower@L1_lower, L1_lower@L1_lower@L1_lower, L1_upper, L1_upper@L1_upper]  
        elif HYPERPARAMS['model'] == 'scnn3':
            shifts = [L1_lower, L1_lower@L1_lower, L1_lower@L1_lower@L1_lower,L1_upper, L1_upper@L1_upper, L1_upper@L1_upper@L1_upper]    
        elif HYPERPARAMS['model'] == 'scnn4':
            shifts = [L1_lower, L1_lower@L1_lower, L1_lower@L1_lower@L1_lower, L1_lower@L1_lower@L1_lower@L1_lower, L1_upper, L1_upper@L1_upper, L1_upper@L1_upper@L1_upper, L1_upper@L1_upper@L1_upper@L1_upper] 
        elif HYPERPARAMS['model'] == 'scnn5':
            shifts = [L1_lower, L1_lower@L1_lower, L1_lower@L1_lower@L1_lower, L1_lower@L1_lower@L1_lower@L1_lower, L1_lower@L1_lower@L1_lower@L1_lower@L1_lower, L1_upper, L1_upper@L1_upper, L1_upper@L1_upper@L1_upper, L1_upper@L1_upper@L1_upper@L1_upper, L1_upper@L1_upper@L1_upper@L1_upper@L1_upper] 
        elif HYPERPARAMS['model'] == 'ebli':
            L1 = L1_lower + L1_upper
            shifts = [L1, L1 @ L1, L1@L1@L1] # L1, L1^2
        elif HYPERPARAMS['model'] == 'bunch' or HYPERPARAMS['model'] =='bunch_2':
            # S_00, S_01, S_01, S_11, S_21, S_12, S_22
            shifts = compute_shift_matrices(B1, B2)
        elif HYPERPARAMS['model'] == 'sccnn1_n' or HYPERPARAMS['model'] =='sccnn1_e' or HYPERPARAMS['model'] =='sccnn2_n' or HYPERPARAMS['model'] =='sccnn2_e':
            shifts = compute_shift_matrices_sccnn1(B1, B2)

        else:
            raise Exception('invalid model type')

    # Build E_lookup for multi-hop training
    e = onp.nonzero(B1.T)[1]
    edges = onp.array_split(e, len(e)/2)
    E, E_lookup = [], {}
    for i, e in enumerate(edges):
        E.append(tuple(e))
        E_lookup[tuple(e)] = i

    # set up neighborhood data
    last_nodes = inputs_all[0][1]

    max_degree = max(G_undir.degree, key=lambda x: x[1])[1]
    nbrhoods_dict = {node: onp.array(list(map(int, G_undir[node]))) for node in
                     map(int, sorted(G_undir.nodes))}
    n_nbrs = onp.array([len(nbrhoods_dict[n]) for n in last_nodes])

    # Bconds function
    nbrhoods = np.array([list(sorted(G_undir[n])) + [-1] * (max_degree - len(G_undir[n])) for n in range(max(G_undir.nodes) + 1)])
    nbrhoods = nbrhoods

    # load prefixes if they exist
    try:
        prefixes = list(np.load('../data/trajectory_data_1hop_' + folder_suffix + '/prefixes.npy', allow_pickle=True))
    except:
        prefixes = [flow_to_path(inputs_all[0][-1][i], E, last_nodes[i]) for i in range(len(last_nodes))]

    B1_jax = np.append(B1, np.zeros((1, B1.shape[1])), axis=0)

    if HYPERPARAMS['flip_edges']:
        B1_jax = B1_jax @ F
        for i in range(len(inputs_all)):
            print(inputs_all[i][-1].shape)
            n_flows, n_edges = inputs_all[i][-1].shape[:2]
            inputs_all[i][-1] = inputs_all[i][-1].reshape((n_flows, n_edges)) @ F
            inputs_all[i][-1] = inputs_all[i][-1].reshape((n_flows, n_edges, 1))

    def Bconds_func(n):
        """
        Returns rows of B1 corresponding to neighbors of node n
        """
        Nv = nbrhoods[n]
        return B1_jax[Nv]

    '''for bunch model option 1'''
    for i in range(len(inputs_all)):
        if HYPERPARAMS['model'] == 'bunch' or  HYPERPARAMS['model'] == 'sccnn1_n' or HYPERPARAMS['model'] == 'sccnn2_n':
            inputs_all[i][0] = nbrhoods
        else:
            inputs_all[i][0] = Bconds_func

            
    if HYPERPARAMS['flip_edges']:
        return inputs_all, y_all, train_mask, test_mask, shifts, G_undir, E_lookup, nbrhoods, n_nbrs, target_nodes_all, prefixes, F 
    else: 
        return inputs_all, y_all, train_mask, test_mask, shifts, G_undir, E_lookup, nbrhoods, n_nbrs, target_nodes_all, prefixes


def train_model():
    """
    Trains a model to predict the next node in each input path (represented as a flow)
    """

    # load dataset
    if HYPERPARAMS['flip_edges']:
        inputs_all, y_all, train_mask, test_mask, shifts, G_undir, E_lookup, nbrhoods, n_nbrs, target_nodes_all, prefixes, F = data_setup(hops=(1,), load=HYPERPARAMS['load_data'], folder_suffix=HYPERPARAMS['data_folder_suffix'])
    else: 
        inputs_all, y_all, train_mask, test_mask, shifts, G_undir, E_lookup, nbrhoods, n_nbrs, target_nodes_all, prefixes = data_setup(hops=(1,), load=HYPERPARAMS['load_data'], folder_suffix=HYPERPARAMS['data_folder_suffix'])
        
    (inputs_1hop, ), (y_1hop, ) = inputs_all, y_all
    #print(len(inputs_1hop), len(y_1hop))
    last_nodes = inputs_1hop[1]

    in_axes = tuple(([None] * len(shifts)) + [None, None, 0, 0])

    # Initialize model
    scone = Scone_GCN(HYPERPARAMS['epochs'], HYPERPARAMS['learning_rate'], HYPERPARAMS['batch_size'], HYPERPARAMS['weight_decay'])

    if HYPERPARAMS['model'] == 'scone':
        model_func = scone_func
    elif HYPERPARAMS['model'] == 'ebli':
        model_func = ebli_func
    elif HYPERPARAMS['model'] == 'bunch':
        model_func = bunch_func
    elif HYPERPARAMS['model'] == 'bunch_2':
        model_func = bunch_func_2
    elif HYPERPARAMS['model'] == 'sccnn1_n':
        model_func = sccnn_func_1_n
    elif HYPERPARAMS['model'] == 'sccnn1_e':
        model_func = sccnn_func_1_e
    elif HYPERPARAMS['model'] == 'sccnn2_n':
        model_func = sccnn_func_2_n
    elif HYPERPARAMS['model'] == 'sccnn2_e':
        model_func = sccnn_func_2_e
    elif HYPERPARAMS['model'] == 'scnn2':
        model_func = scnn_func_2
    elif HYPERPARAMS['model'] == 'scnn12':
        model_func = scnn_func_12
    elif HYPERPARAMS['model'] == 'scnn21':
        model_func = scnn_func_21
    elif HYPERPARAMS['model'] == 'scnn13':
        model_func = scnn_func_13
    elif HYPERPARAMS['model'] == 'scnn31':
        model_func = scnn_func_31
    elif HYPERPARAMS['model'] == 'scnn23':
        model_func = scnn_func_23  
    elif HYPERPARAMS['model'] == 'scnn32':
        model_func = scnn_func_32   
    elif HYPERPARAMS['model'] == 'scnn3':
        model_func = scnn_func_3
    elif HYPERPARAMS['model'] == 'scnn4':
        model_func = scnn_func_4
    elif HYPERPARAMS['model'] == 'scnn5':
        model_func = scnn_func_5
    else:
        raise Exception('invalid model')


    if HYPERPARAMS['model'] == 'scnn2' or HYPERPARAMS['model'] == 'scnn3' or HYPERPARAMS['model'] == 'scnn4' or HYPERPARAMS['model'] == 'scnn5' or HYPERPARAMS['model'] == 'scnn12' or HYPERPARAMS['model'] == 'scnn21' or HYPERPARAMS['model'] == 'scnn13' or HYPERPARAMS['model'] == 'scnn31' or HYPERPARAMS['model'] == 'scnn23' or HYPERPARAMS['model'] == 'scnn32':
        scone.setup_scnn(model_func, HYPERPARAMS['hidden_layers'], HYPERPARAMS['k1_scnn'], HYPERPARAMS['k2_scnn'], shifts, inputs_1hop, y_1hop, in_axes, train_mask, model_type=HYPERPARAMS['model'])
    else:
        scone.setup(model_func, HYPERPARAMS['hidden_layers'], shifts, inputs_1hop, y_1hop, in_axes, train_mask, model_type=HYPERPARAMS['model'])

    if HYPERPARAMS['regional']:
        # Train either on upper region only or all data (synthetic dataset)
        # 0: middle, 1: top, 2: bottom
        train_mask = np.array([1 if i % 3 == 1 else 0 for i in range(len(y_1hop))])
        test_mask = np.array([1 if i % 3 == 2 else 0 for i in range(len(y_1hop))])

    # describe dataset
    if HYPERPARAMS['describe'] == 1:
        print('Graph nodes: {}, edges: {}, avg degree: {}'.format(len(G_undir.nodes), len(G_undir.edges), np.average(np.array([G_undir.degree[node] for node in G_undir.nodes]))))
        print('Training paths: {}, Test paths: {}'.format(train_mask.sum(), test_mask.sum()))
        print('Model: {}'.format(HYPERPARAMS['model']))

    # load a model from file + train it more
    if HYPERPARAMS['load_model']:
        if HYPERPARAMS['regional']:
            scone.weights = onp.load('models/' + HYPERPARAMS['model_name'] + '_' + HYPERPARAMS['model'] + '_' + str(int(HYPERPARAMS['epochs'])) + '_regional' + '.npy', allow_pickle=True)
            print('load successful')
        else:
            scone.weights = onp.load('models/' + HYPERPARAMS['model_name'] + '_' + HYPERPARAMS['model'] + '_' + str(int(HYPERPARAMS['epochs'])) + '.npy', allow_pickle=True)
            print('load successful')
 
        (test_loss, test_acc) = scone.test(inputs_1hop, y_1hop, test_mask, n_nbrs)
        print('test successful')

    elif HYPERPARAMS['reverse']:
        # reverse direction of test flows
        # get the reverse data
        if HYPERPARAMS['flip_edges']:
            rev_flows_in, rev_targets_1hop, rev_targets_2hop, rev_last_nodes = \
                onp.load('../data/trajectory_data_1hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_flows_in.npy'), onp.load('../data/trajectory_data_1hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_targets.npy'), \
                onp.load('../data/trajectory_data_2hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_targets.npy'), onp.load('../data/trajectory_data_1hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_last_nodes.npy') 
            # if flip_edges, need to flip the rev_edges for test acc       
            n_flows, n_edges = rev_flows_in.shape[:2]
            rev_flows_in = rev_flows_in.reshape((n_flows, n_edges)) @ F
            rev_flows_in = rev_flows_in.reshape((n_flows, n_edges, 1))    
            rev_n_nbrs = [len(neighborhood(G_undir, n)) for n in rev_last_nodes]
            rev_inputs_1hop = [inputs_1hop[0], rev_last_nodes, rev_flows_in]
        else:     
            rev_flows_in, rev_targets_1hop, rev_last_nodes = \
                onp.load('../data/trajectory_data_1hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_flows_in.npy'), onp.load('../data/trajectory_data_1hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_targets.npy'), onp.load('../data/trajectory_data_1hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_last_nodes.npy')
            rev_n_nbrs = [len(neighborhood(G_undir, n)) for n in rev_last_nodes]
            rev_inputs_1hop = [inputs_1hop[0], rev_last_nodes, rev_flows_in]  

        train_loss, train_acc, test_loss, test_acc = scone.train(inputs_1hop, y_1hop, train_mask, test_mask, n_nbrs, rev_inputs_1hop, rev_targets_1hop, rev_n_nbrs) 

        try:
            os.mkdir('models')
        except:
            pass

        if HYPERPARAMS['regional']:
            onp.save('models/' + HYPERPARAMS['model_name'] + '_' + HYPERPARAMS['model'] + '_' + str(int(HYPERPARAMS['epochs'])) + '_regional', scone.weights)
        else: 
            onp.save('models/' + HYPERPARAMS['model_name'] + '_' + HYPERPARAMS['model'] + '_' + str(int(HYPERPARAMS['epochs'])), scone.weights)
    else:
        print('Please set the reverse as 1! We also test the reverse prediction performance.')

    # standard experiment # print the final epoch performance
    print('standard test set:')
    scone.test(inputs_1hop, y_1hop, test_mask, n_nbrs)

    # report the final epoch performance for reverse data 
    if HYPERPARAMS['reverse']:
        rev_flows_in, rev_targets_1hop, rev_last_nodes = \
            onp.load('../data/trajectory_data_1hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_flows_in.npy'), onp.load('../data/trajectory_data_1hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_targets.npy'), onp.load('../data/trajectory_data_1hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_last_nodes.npy')
        rev_n_nbrs = [len(neighborhood(G_undir, n)) for n in rev_last_nodes]
        rev_inputs_1hop = [inputs_1hop[0], rev_last_nodes, rev_flows_in] 
        print('Reverse experiment:')
        scone.test(rev_inputs_1hop, rev_targets_1hop, test_mask, rev_n_nbrs)


if __name__ == '__main__':
    train_model()