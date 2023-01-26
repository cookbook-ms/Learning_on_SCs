# Convolutional Learning on Simplicial Complexes
[Semantic Scholar]:https://www.semanticscholar.org/
[Bunch Model]:https://arxiv.org/abs/2012.06010
[SNN]:https://arxiv.org/abs/2010.03633
[PSNN]:https://arxiv.org/abs/2102.10058
[SCNN]:https://ieeexplore.ieee.org/abstract/document/9746017?casa_token=QR4DA8Bju5QAAAAA:xDc2i3Bw7eHtr15KaL1JzWlJEpEnmNHpekuWg5b-aWqVkFnjHCDL9lmPZ3TvG5kdWZAhrCod1w
[GNN]:https://arxiv.org/abs/1606.09375
[CF-SC]:https://arxiv.org/abs/2201.12584
[SCF]:https://arxiv.org/abs/2201.11720
[Benson 2018]: https://arxiv.org/abs/1802.06916

This is a README file for the task of simplex prediction base on the SCCNN in the paper Convolutional Learning on Simplicial Complexes. 

# Data 
The data for this task is generated based on the steps in https://github.com/stefaniaebli/simplicial_neural_networks from [Semantic Scholar]. We use the topology of the generated SC and the simplicial signals, i.e., cochains, which are stored in [data/s2_3_collaboration_complex](./data/s2_3_collaboration_complex). Following codes show how to use the generated data to obtain incidence matrices and simplicial signals. 


```py
cochains_dic = np.load('data/s2_3_collaboration_complex/150250_cochains.npy',allow_pickle=True)
cochains =[list(cochains_dic[i].values()) for i in range(len(cochains_dic))]
# obtain the node, edge and triangle signals
f_node, f_edge, f_tri = cochains[0], cochains[1], cochains[2]
```

```py
boundaries = np.load('data/s2_3_collaboration_complex/150250_boundaries.npy',allow_pickle=True)
boundaries = [boundaries[i].toarray() for i in range(topdim+1)]
# obtain the incidence matrices B1, B2
b1, b2 = boundaries[0], boundaries[1]
```

# Simplex Prediction 
Our method for simplex prediction is a generalization of the [link prediction in based on graph neural networks](https://proceedings.neurips.cc/paper/2018/file/53f0d7c537d99b3824f0f99d62ea2428-Paper.pdf), which is available in https://docs.dgl.ai/en/0.8.x/tutorials/blitz/4_link_predict.html

Consider a 2-simplex prediction, i.e., triangle prediction, performed in a simplicial complex of order two based on our method SCCNN, as an example. This is how [2-simplex_experiments/HOlink_prediction_sccnn_node.py](./2-simplex_experiments/HOlink_prediction_sccnn_node.py) writes. 

## Data Preprocessing
1. Obtain a simplicial complex by fetching the first two incidence matrices $\mathbf{B}_1$ and $\mathbf{B}_2$.  
1. Split the whole triangel set into the positive and negative sets, e.g., based on the values on triangle signals.
2. A split of trianing-validation-test for both positive and negative sets. 
3. Remove the triangles that are not in the positive training set. That is, the simplicial complex used for this task should not contain those triangles not in the positive training set, which can be done by removing the corresponding columns in incidence matrix $\mathbf{B}_2$. 
4. Construct the Hodge Laplacians and the projection matrices. We applied the normalization used in [Bunch Model]. 
5. The positive-negative split and the training-validation-test split are done for the corresponding nodes and edges which form a triangle. For example, for a triangle $t=[i,j,k]$, we can find three edges from $\mathbf{B}_2$, in turn, and we can find three nodes from $\mathbf{B}_1$.   

## Model Training 
In the method of SCCNN of order two, there are nodes, edges and triangles involved. 

1. The inputs on nodes and edges are the given node and edge signals, i.e., 0- and 1-cochains. For the inputs on triangles (positive in the trianing set), we give the input as zeros, as we do not assume any prior knowledges on triangels. 
2. During the training, given above inputs, an SCCNN model is used to learn features on nodes, edges, and triangles. A SCCNN convolution layer is written in [2-simplex_experiments/sccnn_conv.py](./2-simplex_experiments/sccnn_conv.py), which is detailed later on and [2-simplex_experiments/sccnn_model.py](./2-simplex_experiments/sccnn_model.py) stacks them into an SCCNN. 
3. Given the outputs of the SCCNN on three nodes, or three edges for each triangle, we use an MLP to compute a score, which is defined in [2-simplex_experiments/tri_predictor.py](./2-simplex_experiments/tri_predictor.py)
4. A binary cross entropy loss function is used and an AUC can be computed, which are defined in [2-simplex_experiments/tri_predictor.py](./2-simplex_experiments/tri_predictor.py) 


## Model Test 
By computing the ouputs of the SCCNN for the test set and passing them to the MLP, we can obtain the positive and negative scores, which give the AUC performance. 

The trained models are saved in folders [2-simplex_experiments/model_nn_sccnn_node](./2-simplex_experiments/model_nn_sccnn_node) with the specific hyperparemeters as the name, e.g., ```1layers_2order_32features_1rlz``` denoting the number of **intermediate** layers is 1, the convolution orders for all SCFs are 2, and the number of intermediate features is 32 and the data split realization 1.  

<!-- A python file [simplex_pred/auc_extraction.py](./simplex_pred/2-simplex_experiments/auc_extraction.py) can extract the AUC results for all methods with different parameter settings, and [simplex_pred/mean_std_auc.py](./simplex_pred/2-simplex_experiments/mean_std_auc.py) can compute the mean and standard deviation.  -->


## Convolution Layer of an SCCNN
A convolutional layer of an SCCNN is defined in [2-simplex_experiments/sccnn_conv.py](./2-simplex_experiments/sccnn_conv.py). Without loss of generality, we explain how the convolution on edges is performed.

1. Obtain the lower and upper projections from nodes and triangels, where the projection matrices follow [Bunch Model]. 
```py
'''order 1'''
# lower projection 
x1n = self.D2@self.B1.T @torch.inverse(self.D1)@ x0
I1xn = torch.unsqueeze(self.I1@x1n,2)
# identity term (edge input)
I1x = torch.unsqueeze(self.I1@x1,2)
# upper projection
x1p = self.B2 @self.D3@ x2
I1xp = torch.unsqueeze(self.I1@x1p,2)
```
2. Perform simplicial convolutions for each term. 
```py 
if self.k1n > 0: # if lower SCF has an order larger than 0
    X1nl = chebyshev(self.L1l, self.k1n, x1n)
    X1n = torch.cat((I1xn,X1nl),2)
else: 
    X1n = I1xn
```

```py 
if self.k11>0 and self.k12>0: # if both lower and upper conv orders are larger than 0
    X1l = chebyshev(self.L1l, self.k11, x1)
    X1u = chebyshev(self.L1u, self.k12, x1)
    X11 = torch.cat((I1x, X1l, X1u),2)
elif self.k11>0 and self.k12==0: # if only the lower conv order is larger than 0
    X1l = chebyshev(self.L1l, self.k11, x1)
    X11 = torch.cat((I1x, X1l),2)  
elif self.k11==0 and self.k12>0:
    X1u = chebyshev(self.L1u, self.k12, x1)
    X11 = torch.cat((I1x, X1u),2)
else:
    X11 = I1x
```
``` py     
if self.k1p > 0: # if the upper SCF has a conv order larger than 0
    X1pu = chebyshev(self.L1u, self.k1p, x1p)
    X1p = torch.cat((I1xp, X1pu), 2)
else:
    X1p = I1xp
```
3. Concatenate the three convolved results and multiply with the coefficients, i.e., training weights $\mathbf{W}_1$, with the output on edges
```py    
X1 = torch.cat((X1n,X11,X1p),2)
y1 = torch.einsum('nik,iok->no',X1,self.W1)
```

4. Note that the multi-step simplicial convolution is performed in a recursive fashion, defined in [2-simplex_experiments/chebyshev.py](./2-simplex_experiments/chebyshev.py), which avoids matrix-matrix multiplication and admits a linear complexity as studied. 

# To run the code
To run the code for 2-simplex prediction based on SCCNN-Node, one can simply run 
```sh
python ./2-simplex_experiments/sccnn_node.py 
```
Or others, by running 
```sh
python ./2-simplex_experiments/heuristic_method.py # for heuristic methods based on means 
python ./2-simplex_experiments/sccnn_edge.py 
python ./2-simplex_experiments/bunch_node.py 
python ./2-simplex_experiments/bunch_edge.py 
python ./2-simplex_experiments/scnn.py  # for psnn, snn, scnn
python ./2-simplex_experiments/gnn.py
```
Here we include the trained models for SCCNN-Node with $L=2,T=2$ and $F=32$ in the folder [2-simplex_experiments/model_nn_sccnn_node](./2-simplex_experiments/model_nn_sccnn_node) and the text files that record the training losses in the folder [2-simplex_experiments/loss_files](./2-simplex_experiments/loss_files).

We can run [2-simplex_experiments/auc_extraction.py](./2-simplex_experiments/auc_extraction.py) to extract the AUC results for all methods, exported in a file, e.g., [sccnn_node_auc_1layers_32features.txt](./2-simplex_experiments/loss_files/sccnn_node_auc_1layers_32features.txt)

# Other models
Other models include: 
1. Heuristic mean methods [Benson 2018]: implemented in [2-simplex_experiments/heuristic_method.py](./2-simplex_experiments/heuristic_method.py)
1. [Bunch Model]: defined in [2-simplex_experiments/bunch_conv.py](./2-simplex_experiments/bunch_conv.py) and [2-simplex_experiments/bunch_model.py](./2-simplex_experiments/bunch_model.py) stacks the layers
2. GNN and simplicial NNs on one simplex level:
   1. [GNN]: defined in [2-simplex_experiments/gnn_conv.py](./2-simplex_experiments/gnn_conv.py)
   2. [SNN]: defined in [2-simplex_experiments/snn_conv.py](./2-simplex_experiments/snn_conv.py)
   3. [SCNN]: defined in [2-simplex_experiments/scnn_conv.py](./2-simplex_experiments/scnn_conv.py)
   4. [PSNN]: defined in [2-simplex_experiments/psnn_conv.py](./2-simplex_experiments/psnn_conv.py)
3. [CF-SC]: defined in [2-simplex_experiments/sccnn_conv.py](./2-simplex_experiments/sccnn_conv.py) where the nonlinearity is replaced by identity and the stacked in [2-simplex_experiments/sccnn_filter_model](./2-simplex_experiments/sccnn_filter_model.py). 
4. The SCCNN-Node performed in a simplicial complex of order one: defined in [2-simplex_experiments/sccnn_conv.py](./2-simplex_experiments/sccnn_conv.py) too, in class ```sccnn_conv_sc_1``` with a suffix ```sc_1```. The prediction with this SCCNN is in [2-simplex_experiments/ablation_stability/sccnn_node_sc_order1.py](./2-simplex_experiments/ablation_stability/sccnn_node_sc_order1.py). Likewise, SCCNN-Edge of order one, and Bunch-Node and Bunch-Edge of order one are included as well.
5.  Other modified models used for ablation study are also defined in [2-simplex_experiments/sccnn_conv.py](./2-simplex_experiments/sccnn_conv.py). We use suffix to indicate which component is missing. For example, class ```sccnn_conv_no_n_to_e_sc_1``` denotes the case without node to edge contribution in a simplicial complex of order one. These studies can be implemented by [2-simplex_experiments/ablation_stability/sccnn_node_ablation_sc_order1.py](./2-simplex_experiments/ablation_stability/sccnn_node_ablation_sc_order1.py)