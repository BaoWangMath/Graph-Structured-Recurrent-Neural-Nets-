#!/usr/bin/python
'''
Generate features for ST data prediction. 

Author: Bao Wang, Xiyang Luo, Fangbo Zhang
    Department of Mathematics, UCLA
Email: wangbaonj@gmail.com
       xylmath@gmail.com
       fb.zhangsjtu@gmail.com
'''
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import scipy.ndimage
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import h5py
import os

df=pd.read_csv('ILINet.csv')
weeks=df['WEEK']
weeks=weeks.values.reshape((730,10))
weeks=weeks[:579,0]
c1=np.genfromtxt('correlate1.csv',delimiter=',')
test_list=[]

DATA_DIR = '../Data'


def generate_data(fpath): 
    """
    file in fpath should be an n_events x n_postalcode matrix.
    """
    #Events
    df = pd.read_csv(fpath, header=None, sep=',')
    numEventsAll = np.asarray(df).astype('float32').T
    return numEventsAll

def partition_to_three_parts(fpath):
    df = pd.read_csv(fpath, header = None)
    States = range(df.shape[1])
    eventCounts = np.zeros(shape=[len(States)])
    for state_id in States:
        eventCounts[state_id] = np.sum(df[state_id])
    m = np.min(eventCounts)
    M = np.max(eventCounts)
    node_to_type = {}
    for node, ec in enumerate(eventCounts):
        group_num = int(np.floor(float((ec - m) / (M - m + 1e-6)) * 2.))
        if group_num > 1:
            group_num = 1
        node_to_type[node] = group_num
        print node,':',group_num
    return node_to_type

# Xiyang: Modify to one node RNN per US State.
def partition_to_one_node_per_state(fpath):
    df = pd.read_csv(fpath, header = None)
    States = range(df.shape[1])
    num_states = len(States)

    node_to_type = {}
    for node in xrange(num_states):
        node_to_type[node] = node
    return node_to_type    

def load_graph_and_normalize(fpath, threshold=0.025, symmetrize=True, exclude_self=True): 
    weighted_graph = np.loadtxt(fpath, delimiter=',')
    if symmetrize:
        weighted_graph = 0.5 * (weighted_graph + np.transpose(weighted_graph))
    if exclude_self:
        N = len(weighted_graph[0])
        for ind in range(N):
            weighted_graph[ind, ind] = 0
    weighted_graph[weighted_graph < threshold] = 0.0
    # normalize by max degree
    degree = np.sum(weighted_graph, axis=1)
    M = np.max(degree)
    weighted_graph = weighted_graph / M
    graph = {}
    N = weighted_graph.shape[0]
    for n1 in xrange(N):
        graph[n1] = {}
        for n2 in xrange(N):
            if(weighted_graph[n1][n2] > 1e-8):
                graph[n1][n2] = weighted_graph[n1][n2]
    return graph

def ConvertSeriesToMatrix(numEvents, Week_id, len1):
    matrix = []
    # We need to discard the data: 0 ~ len1 -1
    for i in range(len(numEvents) - len1):
        tmp = [Week_id[i+len1]]                         # (feature, label) at the time slot i + len1
        #tmp.append(c1[i-1,2])
        #tmp=[]
        for j in range(i-len1, i):
            tmp.append(numEvents[j])
        # Label
        tmp.append(numEvents[i])
        
        matrix.append(tmp)
    matrix = np.asarray(matrix, dtype=np.float32)
    features = matrix[:, :-1]
    labels = matrix[:, -1]
    features = np.reshape(features, newshape=[features.shape[0], features.shape[1], 1])
    labels = np.reshape(labels, newshape=[labels.shape[0], 1])
    return features, labels
        
        

## US Flu Data ##
class DataReaderUSFlu():
    """Data Reader for US Flu data full graph."""
    def __init__(self):
        self.path = os.path.join(DATA_DIR, 'USStates')
    
    def GenerateNodeFeatures(self):
        numEventsAll = generate_data(os.path.join(self.path, 'activitylevel.csv'))
        scalar1 = MinMaxScaler(feature_range = (0, 1))
        scalar1.fit(numEventsAll.flatten().reshape(-1, 1))
        numEventsAll = scalar1.transform(numEventsAll)
        trX_raw = {}
        trY_raw = {}
        
        # Generate Week ID by another function. TODO
        numWeek = numEventsAll.shape[1]
        Weed_id = np.zeros((numWeek, ))
        for i in range(numWeek):
            Weed_id[i] = (i+1)%52
        
        # Dependence
        numWeek_depend = 2
        for state_id in xrange(numEventsAll.shape[0]):
            features, labels = ConvertSeriesToMatrix(numEventsAll[state_id, : ], Weed_id, numWeek_depend) ## TODO: BW
            
            trX_raw[state_id] = features 
            trY_raw[state_id] = labels
        return trX_raw, trY_raw, scalar1
    
    
    def GenerateNodeClassification(self):
        return partition_to_three_parts(os.path.join(self.path, 'activitylevel.csv'))
        # Xiyang: Change the node partitioning here. You don't need to do anything for the edge partitioning.
        #return partition_to_one_node_per_state(os.path.join(self.path, 'normalized.csv'))
    
    def GenerateGraph(self, threshold=0.01, symmetrize=True, exclude_self=True):
        return load_graph_and_normalize(os.path.join(self.path, 'hhs_graph.csv'), 
            threshold=threshold, symmetrize=symmetrize, exclude_self=exclude_self)
    
    def GenerateNodeIndex(self): 
        return {ind: ind for ind in xrange(51)}