import numpy as np
import os 
from pathlib import Path
path = Path(__file__).parent.absolute()
os.chdir(path)
print(os.getcwd())
'''extract the auc results'''


for model_name in ['sccnn_node','bunch_edge_no_b2']: #'sccnn_node_no_b2','sccnn_node_no_upper','sccnn_node_no_tri','sccnn_node_no_node_to_node','sccnn_node_no_node','sccnn_node_no_lower','sccnn_node_no_edge','sccnn_node_no_b2_no_edge_to_edge','sccnn_node_no_b2_no_edge_to_node','sccnn_node_no_b2_no_node_to_edge','sccnn_node_no_b2_no_node_to_node','sccnn_node_missing_node','sccnn_node_missing_edge','sccnn_node_missing_node_edge'
    print(model_name)
    
    if model_name == 'snn':
        for layers in [0,1,2,3,4]:
            for hidden_features in [16,32]:
                auc_log = open("snn_auc_%dlayers_%dfeatures.txt" %(layers,hidden_features),"w")
                for K in [0,1,2,3,4,5]:
                    for rlz in [1,2,3,4,5,6,7,8,9,10]:
                        losslogf = open("./%s_%dlayers_%dorders_%dfeatures_%drlz.txt" %(model_name,layers,K,hidden_features,rlz),"r")
                        for ln in losslogf:
                            if ln.startswith("AUC:"):
                                auc_log.write("%dlayers_%dorders_%dfeatures_%drlz: %s" %(layers,K,hidden_features,rlz, ln[4:]))
                                auc_log.flush()
                                
    if model_name == 'gnn':
        for layers in [0,1,2,3,4]:
            for hidden_features in [16,32]:
                auc_log = open("gnn_auc_%dlayers_%dfeatures.txt" %(layers,hidden_features),"w")
                for K in [0,1,2,3,4,5]:
                    for rlz in [1,2,3,4,5,6,7,8,9,10]:
                        losslogf = open("./%s_%dlayers_%dorders_%dfeatures_%drlz.txt" %(model_name,layers,K,hidden_features,rlz),"r")
                        for ln in losslogf:
                            if ln.startswith("AUC:"):
                                auc_log.write("%dlayers_%dorders_%dfeatures_%drlz: %s" %(layers,K,hidden_features,rlz, ln[4:]))
                                auc_log.flush()                                

    if model_name == 'psnn':
        for layers in [0,1,2,3,4]:
            for hidden_features in [16,32]:
                auc_log = open("psnn_auc_%dlayers_%dfeatures.txt" %(layers,hidden_features),"w")
                for rlz in [1,2,3,4,5,6,7,8,9,10]:
                    losslogf = open("./%s_%dlayers_%dfeatures_%drlz.txt" %(model_name,layers,hidden_features, rlz), "r")
                    for ln in losslogf:
                        if ln.startswith("AUC:"):
                            auc_log.write("%dlayers_%dfeatures_%drlz: %s" %(layers,hidden_features,rlz, ln[4:]))
                            auc_log.flush()

    if model_name in ['bunch_node','bunch_node_no_b2']:
        for layers in [0,1,2,3,4]:
            for hidden_features in [16,32]:
                auc_log = open("%s_auc_%dlayers_%dfeatures.txt" %(model_name,layers,hidden_features),"w")
                for rlz in [1,2,3,4,5,6,7,8,9,10]:
                    losslogf = open("./%s_%dlayers_%dfeatures_%drlz.txt" %(model_name,layers,hidden_features, rlz), "r")
                    for ln in losslogf:
                        if ln.startswith("AUC:"):
                            auc_log.write("%dlayers_%dfeatures_%drlz: %s" %(layers,hidden_features,rlz, ln[4:]))
                            auc_log.flush()

    if model_name in ['bunch_edge','bunch_edge_no_b2']:
        for layers in [0,1,2,3,4]:
            for hidden_features in [16,32]:
                auc_log = open("%s_auc_%dlayers_%dfeatures.txt" %(model_name,layers,hidden_features),"w")
                for rlz in [1,2,3,4,5,6,7,8,9,10]:
                    losslogf = open("./%s_%dlayers_%dfeatures_%drlz.txt" %(model_name,layers,hidden_features, rlz), "r")
                    for ln in losslogf:
                        if ln.startswith("AUC:"):
                            auc_log.write("%dlayers_%dfeatures_%drlz: %s" %(layers,hidden_features,rlz, ln[4:]))
                            auc_log.flush()

    if model_name == 'scnn':      
        for layers in [0,1,2,3,4]:
            for hidden_features in [16,32]:
                auc_log = open("scnn_auc_%dlayers_%dfeatures.txt" %(layers,hidden_features),"w")
                for K1 in [0,1,2,3,4,5]:
                    for K2 in  [0,1,2,3,4,5]:
                        for rlz in [1,2,3,4,5,6,7,8,9,10]:
                            losslogf = open("./%s_%dlayers_%d_%dorders_%dfeatures_%drlz.txt" %(model_name, layers, K1, K2, hidden_features,rlz), "r")
                            for ln in losslogf:
                                if ln.startswith("AUC:"):
                                    auc_log.write("%dlayers_%d_%dorders_%dfeatures_%drlz: %s" %(layers, K1, K2, hidden_features,rlz, ln[4:]))
                                    auc_log.flush()
                                    
    if model_name == 'sccnn_node':
        for layers in [0,1,2,3,4]:
            for hidden_features in [16,32]:
                auc_log = open("sccnn_node_auc_%dlayers_%dfeatures.txt" %(layers,hidden_features),"w")
                for k in [0,1,2,3,4,5]:
                    for rlz in [1,2,3,4,5,6,7,8,9,10]:
                        losslogf = open("./%s_%dlayers_%dorder_%dfeatures_%drlz.txt" %(model_name,layers,k,hidden_features, rlz), "r")
                        for ln in losslogf:
                            if ln.startswith("AUC:"):
                                auc_log.write("%dlayers_%dorders_%dfeatures_%drlz: %s" %(layers,k,hidden_features,rlz, ln[4:]))
                                auc_log.flush()
                                
    if model_name in  ['sccnn_edge','sccnn_edge_no_b2']:
        for layers in [0,1,2,3,4]:
            for hidden_features in [16,32]:
                auc_log = open("%s_auc_%dlayers_%dfeatures.txt" %(model_name,layers,hidden_features),"w")
                for k in [0,1,2,3,4,5]:
                    for rlz in [1,2,3,4,5,6,7,8,9,10]:
                        losslogf = open("./%s_%dlayers_%dorder_%dfeatures_%drlz.txt" %(model_name,layers,k,hidden_features, rlz), "r")
                        for ln in losslogf:
                            if ln.startswith("AUC:"):
                                auc_log.write("%dlayers_%dorders_%dfeatures_%drlz: %s" %(layers,k,hidden_features,rlz, ln[4:]))
                                auc_log.flush()
                        
    if model_name in ['sccnn_node_missing_node','sccnn_node_missing_edge','sccnn_node_missing_node_edge']:#['sccnn_node_missing_node0.1','sccnn_node_missing_node0.4','sccnn_node_missing_node0.7','sccnn_node_missing_node1']:
        for missing_rate in [0.1,0.4,0.7,1]:
            mn = model_name + str(missing_rate)    
            for layers in [0,1,2,3,4]:
                for hidden_features in [16,32]:
                    auc_log = open("./%s_auc_%dlayers_%dfeatures.txt" %(mn,layers,hidden_features),"w")
                    for k in [0,1,2,3,4,5]:
                        for rlz in [1,2,3,4,5,6,7,8,9,10]:
                            losslogf = open("./%s_%dlayers_%dorder_%dfeatures_%drlz.txt" %(mn,layers,k,hidden_features, rlz), "r")
                            for ln in losslogf:
                                if ln.startswith("AUC:"):
                                    auc_log.write("%dlayers_%dorders_%dfeatures_%drlz: %s" %(layers,k,hidden_features,rlz, ln[4:]))
                                    auc_log.flush()
                                        
    if model_name == 'sccnn_node_ebli':
        for layers in [0,1,2,3,4]:
            for hidden_features in [16,32]:
                auc_log = open("sccnn_node_ebli_auc_%dlayers_%dfeatures.txt" %(layers,hidden_features),"w")
                for k in [0,1,2,3,4,5]:
                    for rlz in [1,2,3,4,5,6,7,8,9,10]:
                        losslogf = open("./%s_%dlayers_%dorder_%dfeatures_%drlz.txt" %(model_name,layers,k,hidden_features, rlz), "r")
                        for ln in losslogf:
                            if ln.startswith("AUC:"):
                                auc_log.write("%dlayers_%dorders_%dfeatures_%drlz: %s" %(layers,k,hidden_features,rlz, ln[4:]))
                                auc_log.flush()
                                
                                
                                
    if model_name in ['sccnn_node_no_b2','sccnn_node_no_upper','sccnn_node_no_tri','sccnn_node_no_node_to_node','sccnn_node_no_node','sccnn_node_no_lower','sccnn_node_no_edge','sccnn_node_no_b2_no_edge_to_edge','sccnn_node_no_b2_no_edge_to_node','sccnn_node_no_b2_no_node_to_edge','sccnn_node_no_b2_no_node_to_node']:
        for layers in [0,1,2,3,4]:
            for hidden_features in [16,32]:
                auc_log = open("./%s_auc_%dlayers_%dfeatures.txt" %(model_name,layers,hidden_features),"w")
                for k in [0,1,2,3,4,5]:
                    for rlz in [1,2,3,4,5,6,7,8,9,10]:
                        losslogf = open("./%s_%dlayers_%dorder_%dfeatures_%drlz.txt" %(model_name,layers,k,hidden_features, rlz), "r")
                        for ln in losslogf:
                            if ln.startswith("AUC:"):
                                auc_log.write("%dlayers_%dorders_%dfeatures_%drlz: %s" %(layers,k,hidden_features,rlz, ln[4:]))
                                auc_log.flush()
                            
                        