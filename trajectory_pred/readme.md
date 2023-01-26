[PSNN]:https://arxiv.org/abs/2102.10058
[Bunch Model]:https://arxiv.org/abs/2012.06010
[SNN]:https://arxiv.org/abs/2010.03633
[PSNN]:https://arxiv.org/abs/2102.10058
[SCNN]:https://ieeexplore.ieee.org/abstract/document/9746017?casa_token=QR4DA8Bju5QAAAAA:xDc2i3Bw7eHtr15KaL1JzWlJEpEnmNHpekuWg5b-aWqVkFnjHCDL9lmPZ3TvG5kdWZAhrCod1w
[GNN]:https://arxiv.org/abs/1606.09375
[CF-SC]:https://arxiv.org/abs/2201.12584
[SCF]:https://arxiv.org/abs/2201.11720
[Schaub et al]:https://arxiv.org/abs/1807.05044

# Trajectory Prediction with SCCNN
This is a README file for the task of trajectory prediction based on the SCCNN proposed in the paper Convolutional Learning on Simplicial Complexes.

Note that the algorithm of trajectory prediction performed in a simplicial complex is introduced by paper, Principled Simplicial Neural Networks [PSNN] for Trajectory Prediction, with its code available at https://github.com/nglaze00/SCoNe_GCN. Based on [PSNN], we applied the SCCNN to this prediction task, together with [SNN], [SCNN] and [Bunch Model] for comparisons.


# Requirements 
Our implementation requires the same packages as in [PSNN], detailed in https://github.com/nglaze00/SCoNe_GCN, including, Python 3; numpy, matplotlib, scipy, networkx, jax, jaxlib, treelib

# Data
We first use the python file [trajectory_analysis/synthetic_data_gen.py](./trajectory_analysis/synthetic_data_gen.py) to generate a synthetic simplicial complexes, together with trajectories, which was proposed in [Schaub et al]. Specifically, we considered the construction with two holes defiend by 400 points, and generated 1000 trajectories based on random walks in edge spaces, following the same procedures as in [PSNN]. We also randomly split these trajectories into 10 training-test sets. One can do so by specifying the rlz_id and random_split_id in the python file, e.g., 
```py
for rlz_id in [2]: #trajectories realiation index
    for random_split_id in [1,2,3,4,5,6,7,8,9,10]: #training-test data spli indices
        folder_suffix = 'working'+str(rlz_id)+'_'+str(random_split_id)  
        generate_dataset(400, 1000, rlz_id, random_split_id, folder_suffix)
```

# Trajectory Prediction Algorithm
We refer to [PSNN] for the details of trajectory prediction algorithm.

# SCCNN Models
As described in the paper, we considered SCCNN-Node and SCCNN-Edge with SCF orders $T=1$ and $T=2$, which are written in the python file [trajectory_analysis/trajectory_experiments.py](./trajectory_analysis/trajectory_experiments.py). Function `sccnn_func_1_n` is for SCCNN-Node of order $T=1$, and `sccnn_func_2_e` is for SCCNN-Edge of order $T=2$. 

To apply the model, one needs to specify the hyperparameters. For example, to use SCCNN-Node of order one, one can set the two parameters `'model'` and `'hidden_layers'` as follows, where tuple `(15,16)` denotes there are 15 shift matrices in total for all node, edge and triangle signals, and 16 features per layer and three tuples make three layers. 
```py
hyperparams = {'model':'sccnn1_n','hidden_layers':[(15,16),(15,16),(15,16)]}
```
Or, one can specify the parameters in the prompt as follows
```sh
python ./trajectory_analysis/trajectory_experiments.py -model sccnn1_n -hidden_layers 15_16_15_16_15_16
```

# Other Models
We also built the code for other models, including 
1. [Bunch Model] with prediction performed through node and edge features. For a three-layer Bunch-Node, it can be applied by settting 
    ```py
    hyperparams={'model':'bunch','hidden_layers':[(7,16),(7,16),(7,16)]}
    ```
    and a Bunch-Edge can be applied by setting the `'model'` as `bunch_2`.
3. [PSNN]: which can be applied by setting 
     ```py
    hyperparams={'model':'scone','hidden_layers':[(3,16),(3,16),(3,16)]}
    ```
4. [SCNN]: for different orders $T_{\rm{d}}$ and $T_{\rm{u}}$, we can set       
    ```py
    hyperparams={'model':'scnn31','hidden_layers':[(3,16),(3,16),(3,16)],'k1_scnn':3,'k2_scnn':1}
    ```
    to obtain perform an [SCNN] of orders $3,1$, where `'hidden_layers'` does not need to change, automated by `'k1_scnn'` and `'k2_scnn'`.
5. [SNN]: for SNN of order $T=3$, we can set 
    ```py
    hyperparams={'model':'ebli','hidden_layers':[(4,16),(4,16),(4,16)]}
    ```

# Integral Lipschitz regularization 
We also applied the integral Lipschitz regularizer to the model [SCNN] and [PSNN] to study if it can help with the stability. Specifically, files [trajectory_analysis/stability_experiment_with_il_rgl.py](./trajectory_analysis/stability_experiment_with_il_rgl.py) and [trajectory_analysis/stability_experiment_no_il_rgl.py](./trajectory_analysis/stability_experiment_no_il_rgl.py) implemented the stability experimetns with and without the regularizer. 

By specifying the `'perturbation'` parameter, one can add perturbations with different norms to the Hodge Laplacians based on the relative perturbation model. For example, we can study the stability of [SCNN] of orders $T_{\rm{d}}=T_{\rm{u}}=3$ with a regularization weight $0.5$ and perturbation level (norm) $0.5$.
```sh
python ./trajectory_analysis/trajectory_stability_experiment.py -model scnn3 -hidden_layers 3_16_3_16_3_16 -weight_il 0.5 -k1_scnn 3 -k2_scnn 3 -perturbation 0.5 
```

Using [trajectory_analysis/il_constant_with_rgl.py](./trajectory_analysis/il_constant_with_rgl.py) and [trajectory_analysis/il_constant_no_rgl.py](./trajectory_analysis/il_constant_no_rgl.py) we can visualize the integral lipschitz property of the trained SCNN with and without the regularizer. 