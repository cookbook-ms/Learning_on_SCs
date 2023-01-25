import sys
sys.path.append('.')
import argparse
import os
from pathlib import Path
path = Path(__file__).parent.absolute()
os.chdir(path)
from statistics import mean

with open('time_sccnn_node_4layers_3order_32features_1rlz.txt','r') as fin:
    data=[x for x in fin.read().split('\n')]
    

x = map(float, data[:-1])
x = list(x)
x = x[1:]
print(sum(x)/len(x))
