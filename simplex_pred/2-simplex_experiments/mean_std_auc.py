import numpy as np
import os 
from pathlib import Path

path = Path(__file__).parent.absolute()
os.chdir(path)
cwd = os.getcwd()
print(cwd)

'''compute the mean and std of the acu results'''

'''
psnn
'''
for layers in range(0,5):
    for features in [16,32]:
        psnn_auc=[]
        with open("psnn_auc_%dlayers_%dfeatures.txt"%(layers,features), "r") as filestream:
            for line in filestream:
                currentline = line.split(":")
                a = float(currentline[1][2:-1])
                psnn_auc.append(a)
                
        psnn_auc = np.array(psnn_auc)
        psnn_auc = np.reshape(psnn_auc,(10,-1),'F')

        psnn_auc_mean = np.mean(psnn_auc,0)
        psnn_auc_std = np.std(psnn_auc,0)
        # print('psnn:','\n',(psnn_auc_mean), (psnn_auc_std))
        print('psnn:','\n',np.round(psnn_auc_mean,2), np.round(psnn_auc_std,3))

'''
snn-1,-2,-3,-4,-5
'''
for layers in range(0,5):
    print(layers)
    for features in [16,32]:  
        snn_auc = []
        with open("snn_auc_%dlayers_%dfeatures.txt" %(layers,features), "r") as filestream:
            for line in filestream:
                currentline = line.split(":")
                a = float(currentline[1][2:-1])
                snn_auc.append(a)
            
        snn_auc = np.array(snn_auc)
        snn_auc = np.reshape(snn_auc,(10,-1),'F')

        snn_auc_mean = np.mean(snn_auc,0)
        snn_auc_std = np.std(snn_auc,0)
        print('snn:',np.round(snn_auc_mean,2),'\n',np.round(snn_auc_std,3))
        
'''gnn'''       
for layers in range(0,5):
    print(layers)
    for features in [16,32]:  
        gnn_auc = []
        with open("gnn_auc_%dlayers_%dfeatures.txt" %(layers,features), "r") as filestream:
            for line in filestream:
                currentline = line.split(":")
                a = float(currentline[1][2:-1])
                gnn_auc.append(a)
            
        gnn_auc = np.array(gnn_auc)
        gnn_auc = np.reshape(gnn_auc,(10,-1),'F')

        gnn_auc_mean = np.mean(gnn_auc,0)
        gnn_auc_std = np.std(gnn_auc,0)
        print('gnn:',np.round(gnn_auc_mean,2),'\n',np.round(gnn_auc_std,3))


'''
scnn
'''

for layers in range(0,5):
    print(layers)
    for features in [16,32]:  
        scnn_auc = []
        with open("scnn_auc_%dlayers_%dfeatures.txt" %(layers,features), "r") as filestream:
            for line in filestream:
                currentline = line.split(":")
                a = float(currentline[1][2:-1])
                scnn_auc.append(a)
            
        scnn_auc = np.array(scnn_auc)
        scnn_auc = np.reshape(scnn_auc,(10,6,6),'F')
        scnn_auc_mean = np.mean(scnn_auc,0)
        scnn_auc_std = np.std(scnn_auc,0)
        print('scnn:',np.round(scnn_auc_mean,2),'\n',np.round(scnn_auc_std,3))
        
        
'''
bunch_node
'''
for layers in range(0,5):
    for features in [16,32]:
        bunch_node_auc=[]
        with open("bunch_node_auc_%dlayers_%dfeatures.txt"%(layers,features), "r") as filestream:
            for line in filestream:
                currentline = line.split(":")
                a = float(currentline[1][2:-1])
                bunch_node_auc.append(a)
                
        bunch_node_auc = np.array(bunch_node_auc)
        bunch_node_auc = np.reshape(bunch_node_auc,(10,-1),'F')

        bunch_node_auc_mean = np.mean(bunch_node_auc,0)
        bunch_node_auc_std = np.std(bunch_node_auc,0)
        print('bunch-node:','\n',np.round(bunch_node_auc_mean,2), np.round(bunch_node_auc_std,3))

'''
bunch_edge
'''
for layers in range(0,5):
    for features in [16,32]:
        bunch_edge_auc=[]
        with open("bunch_edge_auc_%dlayers_%dfeatures.txt"%(layers,features), "r") as filestream:
            for line in filestream:
                currentline = line.split(":")
                a = float(currentline[1][2:-1])
                bunch_edge_auc.append(a)
                
        bunch_edge_auc = np.array(bunch_edge_auc)
        bunch_edge_auc = np.reshape(bunch_edge_auc,(10,-1),'F')

        bunch_edge_auc_mean = np.mean(bunch_edge_auc,0)
        bunch_edge_auc_std = np.std(bunch_edge_auc,0)
        print('bunch-edge:','\n',np.round(bunch_edge_auc_mean,2), np.round(bunch_edge_auc_std,3))
        
'''
sccnn node
'''

for layers in range(0,5):
    print(layers)
    for features in [16,32]:  
        sccnn_node_auc = []
        with open("sccnn_node_auc_%dlayers_%dfeatures.txt" %(layers,features), "r") as filestream:
            for line in filestream:
                currentline = line.split(":")
                a = float(currentline[1][2:-1])
                sccnn_node_auc.append(a)
            
        sccnn_node_auc = np.array(sccnn_node_auc)
        sccnn_node_auc = np.reshape(sccnn_node_auc,(10,-1),'F')
        sccnn_node_auc_mean = np.mean(sccnn_node_auc,0)
        sccnn_node_auc_std = np.std(sccnn_node_auc,0)
        print('sccnn node:',np.round(sccnn_node_auc_mean,4),'\n',np.round(sccnn_node_auc_std,3))
        
        
'''
sccnn edge
'''
for model_name in ['sccnn_edge_no_b2']:
    print(model_name)
    for layers in range(0,5):
        print(layers)
        for features in [16,32]:  
            sccnn_edge_auc = []
            with open("%s_auc_%dlayers_%dfeatures.txt" %(model_name,layers,features), "r") as filestream:
                for line in filestream:
                    currentline = line.split(":")
                    a = float(currentline[1][2:-1])
                    sccnn_edge_auc.append(a)
                
            sccnn_edge_auc = np.array(sccnn_edge_auc)
            sccnn_edge_auc = np.reshape(sccnn_edge_auc,(10,-1),'F')
            sccnn_edge_auc_mean = np.mean(sccnn_edge_auc,0)
            sccnn_edge_auc_std = np.std(sccnn_edge_auc,0)
            print('sccnn edge:',np.round(sccnn_edge_auc_mean,3),'\n',np.round(sccnn_edge_auc_std,3))


'''
for missing input
'''
for model_name in ['bunch_node_no_b2','bunch_edge_no_b2']:#'sccnn_node_missing_node','sccnn_node_missing_edge','sccnn_node_missing_node_edge'
 #   for missing_rate in [0.1,0.4,0.7,1]:
 
    mn = model_name #+ str(missing_rate) 
    print(mn)
    for layers in range(0,5):
        print(layers)
        for features in [16,32]:  
            sccnn_node_auc = []
            with open("./%s_auc_%dlayers_%dfeatures.txt" %(mn,layers,features), "r") as filestream:
                for line in filestream:
                    currentline = line.split(":")
                    a = float(currentline[1][2:-1])
                    sccnn_node_auc.append(a)
                
            sccnn_node_auc = np.array(sccnn_node_auc)
            sccnn_node_auc = np.reshape(sccnn_node_auc,(10,-1),'F')
            sccnn_node_auc_mean = np.mean(sccnn_node_auc,0)
            sccnn_node_auc_std = np.std(sccnn_node_auc,0)
            print('sccnn node:',np.round(sccnn_node_auc_mean,3),'\n',np.round(sccnn_node_auc_std,3))


'''
sccnn node -- ablation study 
'''
for model_name in ['sccnn_node_no_b2','sccnn_node_no_upper','sccnn_node_no_tri','sccnn_node_no_node_to_node','sccnn_node_no_node','sccnn_node_no_lower','sccnn_node_no_edge','sccnn_node_no_b2_no_edge_to_edge','sccnn_node_no_b2_no_edge_to_node','sccnn_node_no_b2_no_node_to_edge','sccnn_node_no_b2_no_node_to_node']:
    print(model_name)
    for layers in range(0,5):
        print(layers)
        for features in [16,32]:  
            sccnn_node_auc = []
            with open("%s_auc_%dlayers_%dfeatures.txt" %(model_name,layers,features), "r") as filestream:
                for line in filestream:
                    currentline = line.split(":")
                    a = float(currentline[1][2:-1])
                    sccnn_node_auc.append(a)
                
            sccnn_node_auc = np.array(sccnn_node_auc)
            sccnn_node_auc = np.reshape(sccnn_node_auc,(10,-1),'F')
            sccnn_node_auc_mean = np.mean(sccnn_node_auc,0)
            sccnn_node_auc_std = np.std(sccnn_node_auc,0)
            print('sccnn node:',np.round(sccnn_node_auc_mean,3),'\n',np.round(sccnn_node_auc_std,3))