'''
Study the stability 
'''


from cmath import tan
import os, sys
import numpy as onp
from numpy import linalg as la
import jax.numpy as np
from jax.scipy.special import logsumexp
from pathlib import Path
path = Path(__file__).parent.absolute()
os.chdir(path)

try:
    from trajectory_analysis.bunch_model_matrices import compute_shift_matrices
    from trajectory_analysis.synthetic_data_gen import load_dataset, generate_dataset, neighborhood, conditional_incidence_matrix, flow_to_path
    from trajectory_analysis.scnn_trajectory_model_stability import Scone_GCN
except Exception:
    from bunch_model_matrices import compute_shift_matrices
    from synthetic_data_gen import load_dataset, generate_dataset, neighborhood, conditional_incidence_matrix, flow_to_path
    from scnn_trajectory_model_stability import Scone_GCN

def hyperparams():
    """
    Parse hyperparameters from command line

    For hidden_layers, input [(3, 8), (3, 8)] as 3_8_3_8
    """
    args = sys.argv
    hyperparams = {'model': 'scnn12',
                   'epochs': 1000,
                   'learning_rate': 0.001,
                   'weight_decay': 0.00005,
                   'weight_il': 0.0005,
                   'batch_size': 100,
                   'hidden_layers': [(3, 16)], 
                   'k1_scnn': 1,
                   'k2_scnn': 2,
                   'describe': 1,
                   'reverse': 1,
                   'load_data': 1,
                   'load_model': 0,
                   'normalized': 1,
                   'markov': 0,
                   'model_name': 'model',
                   'regional': 0,
                   'flip_edges': 0,
                   'data_folder_suffix': 'working',
                   'multi_graph': '',                   
                   'perturbation': 0.001,          # perturbations 
    #    array([0.001     , 0.00199526, 0.00398107, 0.00794328, 0.01584893,
    #    0.03162278, 0.06309573, 0.12589254, 0.25118864, 0.50118723,
    #    1.        ])
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
    #print('#weights:',len(weights),'#layers:',n_layers)
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = flow
    #print(S_lower.size)
    for i in range(int(n_layers)):
        cur_out = cur_out @ weights[i * 3] \
                  + S_lower @ cur_out @ weights[i*3 + 1] \
                  + S_upper @ cur_out @ weights[i*3 + 2]

        cur_out = tanh(cur_out)

    logits = Bcond_func(last_node) @ cur_out @ weights[-1]
    #print(logits)
    return logits - logsumexp(logits), cur_out # log of the softmax function 

def scnn_func_1(weights, S_lower, S_upper, Bcond_func, last_node, flow, k1=HYPERPARAMS['k1_scnn'], k2=HYPERPARAMS['k2_scnn']):
    """
    Forward pass of the SCoNe model with variable number of layers
    """
    n_layers = (len(weights) - 1) / (1+k1+k2) # k1=k2=1
    n_k = 1+k1+k2
    #print('#weights:',len(weights),'#layers:',n_layers)
    assert n_layers % 1 == 0, 'wrong number of weights'
    cur_out = flow
    for i in range(int(n_layers)):
        cur_out = cur_out @ weights[i*n_k] \
                  + S_lower @ cur_out @ weights[i*n_k + 1] \
                  + S_upper @ cur_out @ weights[i*n_k + 2] 

        cur_out = tanh(cur_out)

    logits = Bcond_func(last_node) @ cur_out @ weights[-1]
    return logits - logsumexp(logits), cur_out

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
    return logits - logsumexp(logits), cur_out

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
    return logits - logsumexp(logits), cur_out


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
    return logits - logsumexp(logits), cur_out

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
    return logits - logsumexp(logits), cur_out


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
    return logits - logsumexp(logits), cur_out


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
    return logits - logsumexp(logits), cur_out

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
    return logits - logsumexp(logits), cur_out


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
    return logits - logsumexp(logits), cur_out

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
    return logits - logsumexp(logits), cur_out


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
    return logits - logsumexp(logits), cur_out


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
    return logits - logsumexp(logits), cur_out

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

        cur_out = [relu(c) for c in next_out]

    nodes_out = cur_out[0] # use the last layer output on the node level as the final output 
    # values at nbrs of last node
    logits = nodes_out[nbrhoods[last_node]]
    return logits - logsumexp(logits)


def data_setup(hops=(1,), load=True, folder_suffix='schaub'):
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

        # perturbations 
        epsilon_lower = HYPERPARAMS['perturbation'] 
        print(np.linalg.norm(L1_lower))  
        E1_lower_rlz = onp.mean([onp.random.randn(L1_lower.shape[0],L1_lower.shape[1]) for k in range(10)],0)
        E1_lower = E1_lower_rlz/la.norm(E1_lower_rlz)*epsilon_lower
        print(np.linalg.norm(E1_lower)) 

        epsilon_upper = epsilon_lower
        print(np.linalg.norm(L1_upper))
        E1_upper_rlz = onp.mean([onp.random.randn(L1_upper.shape[0],L1_upper.shape[1]) for k in range(10)],0)
        E1_upper = E1_upper_rlz/la.norm(E1_upper_rlz)*epsilon_upper
        print(np.linalg.norm(E1_upper)) 

        L1_lower_perturbed = L1_lower + E1_lower @ L1_lower + L1_lower @ E1_lower
        L1_upper_perturbed = L1_upper + E1_upper @ L1_upper + L1_upper @ E1_upper

        print(la.norm(L1_lower_perturbed), la.norm(L1_upper_perturbed))
        print(L1_lower_perturbed.shape, L1_upper_perturbed.shape)

        if HYPERPARAMS['model'] == 'scone':
            shifts = [L1_lower, L1_upper]
            perturbed_shifts = [L1_lower_perturbed, L1_upper_perturbed]
            # shifts = [L1_lower, L1_lower]

        elif HYPERPARAMS['model'] == 'scnn1':
            shifts = [L1_lower, L1_upper]
            perturbed_shifts = [L1_lower_perturbed, L1_upper_perturbed]

        elif HYPERPARAMS['model'] == 'scnn2':
            # shifts = [la.matrix_power(L1_lower,i+1) for i in range(HYPERPARAMS['k1_scnn'])]
            # shifts.append(la.matrix_power(L1_upper,i+1) for i in range(HYPERPARAMS['k2_scnn']))
            shifts = [L1_lower, L1_lower@L1_lower, L1_upper, L1_upper@L1_upper]
            perturbed_shifts = [L1_lower_perturbed, L1_lower_perturbed@L1_lower_perturbed, L1_upper_perturbed, L1_upper_perturbed@L1_upper_perturbed]
            #L1 = L1_lower + L1_upper
            #shifts = [L1, L1 @ L1, L1@L1@L1,L1@L1@L1@L1] # test order 4 ebli _func 

        elif HYPERPARAMS['model'] == 'scnn12':
            shifts = [L1_lower, L1_upper, L1_upper@L1_upper]
            perturbed_shifts = [L1_lower_perturbed, L1_upper_perturbed, L1_upper_perturbed@L1_upper_perturbed]
        
        elif HYPERPARAMS['model'] == 'scnn21':
            shifts = [L1_lower, L1_lower@L1_lower, L1_upper]
            perturbed_shifts = [L1_lower_perturbed, L1_lower_perturbed@L1_lower_perturbed, L1_upper_perturbed]

        elif HYPERPARAMS['model'] == 'scnn13':
            shifts = [L1_lower, L1_upper, L1_upper@L1_upper, L1_upper@L1_upper@L1_upper]
            perturbed_shifts = [L1_lower_perturbed, L1_upper_perturbed, L1_upper_perturbed@L1_upper_perturbed, L1_upper_perturbed@L1_upper_perturbed@L1_upper_perturbed] 

        elif HYPERPARAMS['model'] == 'scnn31':
            shifts = [L1_lower, L1_lower@L1_lower, L1_lower@L1_lower@L1_lower, L1_upper]
            perturbed_shifts = [L1_lower_perturbed, L1_lower_perturbed@L1_lower_perturbed, L1_lower_perturbed@L1_lower_perturbed@L1_lower_perturbed, L1_upper_perturbed] 

        elif HYPERPARAMS['model'] == 'scnn23':
            shifts = [L1_lower, L1_lower@L1_lower, L1_upper, L1_upper@L1_upper, L1_upper@L1_upper@L1_upper]
            perturbed_shifts = [L1_lower_perturbed, L1_lower_perturbed@L1_lower_perturbed, L1_upper_perturbed, L1_upper_perturbed@L1_upper_perturbed, L1_upper_perturbed@L1_upper_perturbed@L1_upper_perturbed]   

        elif HYPERPARAMS['model'] == 'scnn32':
            shifts = [L1_lower, L1_lower@L1_lower, L1_lower@L1_lower@L1_lower, L1_upper, L1_upper@L1_upper]  
            perturbed_shifts = [L1_lower_perturbed, L1_lower_perturbed@L1_lower_perturbed, L1_lower_perturbed@L1_lower_perturbed@L1_lower_perturbed, L1_upper_perturbed, L1_upper_perturbed@L1_upper_perturbed]  
           
        elif HYPERPARAMS['model'] == 'scnn3':
            shifts = [L1_lower, L1_lower@L1_lower, L1_lower@L1_lower@L1_lower,L1_upper, L1_upper@L1_upper, L1_upper@L1_upper@L1_upper]    
            perturbed_shifts = [L1_lower_perturbed, L1_lower_perturbed@L1_lower_perturbed, L1_lower_perturbed@L1_lower_perturbed@L1_lower_perturbed, L1_upper_perturbed, L1_upper_perturbed@L1_upper_perturbed, L1_upper_perturbed@L1_upper_perturbed@L1_upper_perturbed]    

        elif HYPERPARAMS['model'] == 'scnn4':
            shifts = [L1_lower, L1_lower@L1_lower, L1_lower@L1_lower@L1_lower, L1_lower@L1_lower@L1_lower@L1_lower, L1_upper, L1_upper@L1_upper, L1_upper@L1_upper@L1_upper, L1_upper@L1_upper@L1_upper@L1_upper] 
            perturbed_shifts = [L1_lower_perturbed, L1_lower_perturbed@L1_lower_perturbed, L1_lower_perturbed@L1_lower_perturbed@L1_lower_perturbed, L1_lower_perturbed@L1_lower_perturbed@L1_lower_perturbed@L1_lower_perturbed, L1_upper_perturbed, L1_upper_perturbed@L1_upper_perturbed, L1_upper_perturbed@L1_upper_perturbed@L1_upper_perturbed, L1_upper_perturbed@L1_upper_perturbed@L1_upper_perturbed@L1_upper_perturbed] 

        elif HYPERPARAMS['model'] == 'scnn5':
            shifts = [L1_lower, L1_lower@L1_lower, L1_lower@L1_lower@L1_lower, L1_lower@L1_lower@L1_lower@L1_lower, L1_lower@L1_lower@L1_lower@L1_lower@L1_lower, L1_upper, L1_upper@L1_upper, L1_upper@L1_upper@L1_upper, L1_upper@L1_upper@L1_upper@L1_upper, L1_upper@L1_upper@L1_upper@L1_upper@L1_upper] 
            perturbed_shifts = [L1_lower_perturbed, L1_lower_perturbed@L1_lower_perturbed, L1_lower_perturbed@L1_lower_perturbed@L1_lower_perturbed, L1_lower_perturbed@L1_lower_perturbed@L1_lower_perturbed@L1_lower_perturbed, L1_lower_perturbed@L1_lower_perturbed@L1_lower_perturbed@L1_lower_perturbed@L1_lower_perturbed, L1_upper_perturbed, L1_upper_perturbed@L1_upper_perturbed, L1_upper_perturbed@L1_upper_perturbed@L1_upper_perturbed, L1_upper_perturbed@L1_upper_perturbed@L1_upper_perturbed@L1_upper_perturbed, L1_upper_perturbed@L1_upper_perturbed@L1_upper_perturbed@L1_upper_perturbed@L1_upper_perturbed] 
            
        elif HYPERPARAMS['model'] == 'ebli':
            L1 = L1_lower + L1_upper
            shifts = [L1, L1 @ L1, L1@L1@L1] # L1, L1^2

        elif HYPERPARAMS['model'] == 'bunch':
            # S_00, S_01, S_01, S_11, S_21, S_12, S_22
            shifts = compute_shift_matrices(B1, B2)

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

    for i in range(len(inputs_all)):
        if HYPERPARAMS['model'] != 'bunch':
            inputs_all[i][0] = Bconds_func
        else:
            inputs_all[i][0] = nbrhoods
            
    if HYPERPARAMS['flip_edges']:
        return inputs_all, y_all, train_mask, test_mask, shifts, perturbed_shifts, G_undir, E_lookup, nbrhoods, n_nbrs, target_nodes_all, prefixes, F, L1_lower, L1_upper 
    else: 
        return inputs_all, y_all, train_mask, test_mask, shifts, perturbed_shifts, G_undir, E_lookup, nbrhoods, n_nbrs, target_nodes_all, prefixes, L1_lower, L1_upper 
##
def train_model():
    """
    Trains a model to predict the next node in each input path (represented as a flow)
    """

    # load dataset
    if HYPERPARAMS['flip_edges']:
        inputs_all, y_all, train_mask, test_mask, shifts, perturbed_shifts, G_undir, E_lookup, nbrhoods, n_nbrs, target_nodes_all, prefixes, F, L1_lower, L1_upper  = data_setup(hops=(1,2), load=HYPERPARAMS['load_data'], folder_suffix=HYPERPARAMS['data_folder_suffix'])
    else: 
        inputs_all, y_all, train_mask, test_mask, shifts, perturbed_shifts, G_undir, E_lookup, nbrhoods, n_nbrs, target_nodes_all, prefixes, L1_lower, L1_upper  = data_setup(hops=(1,2), load=HYPERPARAMS['load_data'], folder_suffix=HYPERPARAMS['data_folder_suffix'])
        
    (inputs_1hop, inputs_2hop), (y_1hop, y_2hop) = inputs_all, y_all
    #print(len(inputs_1hop), len(y_1hop))
    last_nodes = inputs_1hop[1]
    print(len(shifts),len(perturbed_shifts))
    print(shifts[0].shape,perturbed_shifts[0].shape)
    in_axes = tuple(([None] * len(shifts)) + [None, None, 0, 0])

    # Initialize model
    scone = Scone_GCN(HYPERPARAMS['epochs'], HYPERPARAMS['learning_rate'], HYPERPARAMS['batch_size'], HYPERPARAMS['weight_decay'], HYPERPARAMS['weight_il'], HYPERPARAMS['perturbation'])

    if HYPERPARAMS['model'] == 'scone':
        model_func = scone_func
    elif HYPERPARAMS['model'] == 'ebli':
        model_func = ebli_func
    elif HYPERPARAMS['model'] == 'bunch':
        model_func = bunch_func
    elif HYPERPARAMS['model'] == 'scnn1':
        model_func = scnn_func_1
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


    if HYPERPARAMS['model'] == 'scone' or HYPERPARAMS['model'] == 'scnn1' or HYPERPARAMS['model'] == 'scnn2' or HYPERPARAMS['model'] == 'scnn3' or HYPERPARAMS['model'] == 'scnn4' or HYPERPARAMS['model'] == 'scnn5' or HYPERPARAMS['model'] == 'scnn12' or HYPERPARAMS['model'] == 'scnn21' or HYPERPARAMS['model'] == 'scnn13' or HYPERPARAMS['model'] == 'scnn31' or HYPERPARAMS['model'] == 'scnn23' or HYPERPARAMS['model'] == 'scnn32':
        scone.setup_scnn(model_func, HYPERPARAMS['hidden_layers'], HYPERPARAMS['k1_scnn'], HYPERPARAMS['k2_scnn'], shifts, perturbed_shifts, inputs_1hop, y_1hop, in_axes, train_mask, model_type=HYPERPARAMS['model'])
    else:
        scone.setup(model_func, HYPERPARAMS['hidden_layers'], shifts, perturbed_shifts, inputs_1hop, y_1hop, in_axes, train_mask, model_type=HYPERPARAMS['model'])

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
            # rev_n_nbrs = [len(neighborhood(G_undir, n)) for n in rev_last_nodes]
            # rev_inputs_1hop = [inputs_1hop[0], rev_last_nodes, rev_flows_in] 
        else:     
            rev_flows_in, rev_targets_1hop, rev_targets_2hop, rev_last_nodes = \
                onp.load('../data/trajectory_data_1hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_flows_in.npy'), onp.load('../data/trajectory_data_1hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_targets.npy'), \
                onp.load('../data/trajectory_data_2hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_targets.npy'), onp.load('../data/trajectory_data_1hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_last_nodes.npy')
            rev_n_nbrs = [len(neighborhood(G_undir, n)) for n in rev_last_nodes]
            rev_inputs_1hop = [inputs_1hop[0], rev_last_nodes, rev_flows_in]  

        train_loss, train_acc, test_loss, test_acc = scone.train(inputs_1hop, y_1hop, train_mask, test_mask, n_nbrs, rev_inputs_1hop, rev_targets_1hop, rev_n_nbrs, L1_lower, L1_upper, HYPERPARAMS['k1_scnn'], HYPERPARAMS['k2_scnn']) 

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
    scone.test(inputs_1hop, y_1hop, test_mask, n_nbrs, L1_lower, L1_upper, HYPERPARAMS['k1_scnn'], HYPERPARAMS['k2_scnn'])
    
    # train_2target, test_2target = scone.two_target_accuracy(shifts, inputs_1hop, y_1hop, train_mask, n_nbrs), scone.two_target_accuracy(shifts, inputs_1hop, y_1hop, test_mask, n_nbrs)

    # print('2-target accs:', train_2target, test_2target)

    # report the final epoch performance for reverse data 
    if HYPERPARAMS['reverse']:
        rev_flows_in, rev_targets_1hop, rev_targets_2hop, rev_last_nodes = \
            onp.load('../data/trajectory_data_1hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_flows_in.npy'), onp.load('../data/trajectory_data_1hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_targets.npy'), \
            onp.load('../data/trajectory_data_2hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_targets.npy'), onp.load('../data/trajectory_data_1hop_' + HYPERPARAMS['data_folder_suffix'] + '/rev_last_nodes.npy')
        rev_n_nbrs = [len(neighborhood(G_undir, n)) for n in rev_last_nodes]
        rev_inputs_1hop = [inputs_1hop[0], rev_last_nodes, rev_flows_in] 
        print('Reverse experiment:')
        scone.test(rev_inputs_1hop, rev_targets_1hop, test_mask, rev_n_nbrs, L1_lower, L1_upper, HYPERPARAMS['k1_scnn'], HYPERPARAMS['k2_scnn'])


if __name__ == '__main__':
    train_model()