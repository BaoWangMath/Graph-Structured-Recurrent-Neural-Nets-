# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
This code is used to predict the flu for all states in US using sRNN.
Previous 10 weeks data is used as features.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import scipy.ndimage
import csv
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
import sys
import json
import srnn_tf
from srnn_tf import InverseTransform, TrainValSplit
import FeatureGeneration

DATA_DIR = '../Data'
TRAIN_EPOCHS = 110
BATCH_SIZE = 10  # In total, we have < 400 data

GPU_MEMORY_FRACTION = 1.0  # adjust this so that allocated memory is 5G

# Choices for DATA_TYPE
DATA_TYPE = 'USStates'

# Choice for MODEL_TYPE:
# - SRNNUndirected: undirected srnn
# - SRNNDirected: directed srnn
MODEL_TYPE = 'SRNNUndirected'

#Resize the image
def img_enlarge(img, factor = 2.0, order = 2):
    return scipy.ndimage.zoom(img, factor, order = order)

def node_cell():
    """
    Definition for NodeRNN.    
    """
    num_units = 64
    return tf.contrib.rnn.BasicLSTMCell(num_units=num_units)

def edge_cell():
    """
    Definition for EdgeRNN.
    """
    num_units = 64
    return tf.contrib.rnn.BasicLSTMCell(num_units=num_units)

def edge_cell_2():
    """
    Define lstm cell(s) for edge rnn
    """
    num_units = [40]
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.BasicLSTMCell(num_units=nu) for nu in num_units])
    return stacked_lstm

# Xiyang: This is the four layer lstm. 
def four_layer_lstm():
    """
    Define lstm cell(s) for edge rnn
    """
    # Xiyang: Specify the output dimension of each layer of the lstm here.
    # For node RNN, the final Dense(1) is already in the code. 
    # If you want to do that for the edge RNNs as well, you need to add tf.layer.dense to the edge output.
    num_units = [10,40,10]
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.BasicLSTMCell(num_units=nu) for nu in num_units])
    return stacked_lstm    

#RNN predictor
def RNNPrediction(trX_raw, trY_raw, scalar1, graph, node_to_type, srnn_model):    
    with tf.Graph().as_default() as g: 
        sql = trX_raw[0].shape[1]
        node_input_dims = trX_raw[0].shape[2]
        edge_input_dims = node_input_dims
        output_dims = trY_raw[0].shape[1]
        
        # Xiyang: Specify the node and edge cells here. 
        node_cell_fn = four_layer_lstm
        edge_cell_fn = edge_cell_2
        
        model = srnn_model(
            graph, node_to_type, node_input_dims=node_input_dims, edge_input_dims=edge_input_dims, 
            output_dims=output_dims, seq_length=sql, node_cell=node_cell_fn, 
            edge_cell=edge_cell_fn, use_dropout=True, dropout_rate=0.2)
        model.build_model()
        trX, trY = model.add_edge_features(trX_raw, trY_raw)
        print "Number of nodes : %d" % len(trX.keys())
        cls = np.array([node_to_type[t] for t in node_to_type])
        cls_list = np.unique(cls)
        print "Total number of classes : %d" % len(cls_list)
        for cl in cls_list: 
            print "Num Nodes in Class %d : %d" % (cl, np.sum(cls == cl))
        print ""
        print "Node feature shape = ", trX[0]['%d_input' % model.node_to_type[0]].shape
        for en in trX[0].keys(): 
            if en !=  '%d_input' % model.node_to_type[0]: 
                print "Edge Feature Shape = ", trX[0][en].shape
                break
        print "Number of nodes in graph: %d" % len(graph.keys())
        print "Total number of training epochs: %d" % TRAIN_EPOCHS
        trX_train, trX_test, trY_train, trY_test = TrainValSplit(trX, trY, ratio=0.75)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEMORY_FRACTION)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess: 
            model.fit_model(sess=sess, trX=trX_train, trY=trY_train, batch_size=BATCH_SIZE, validation_batch_size=BATCH_SIZE,
                max_iter=TRAIN_EPOCHS, learning_rate=0.001, num_epochs_per_summary=1,
                validation_split=0.1, save_model_epoch=20, save_model_dir=DUMP_PATH, sample_rate=1.0)
    
            # predict
            trainPredict = model.predict_node(sess, trX_train)
            testPredict = model.predict_node(sess, trX_test)
    
    #Invert the prediction
    trainPredict = InverseTransform(trainPredict, scalar1)
    testPredict = InverseTransform(testPredict, scalar1)
    
    train = InverseTransform(trY_train, scalar1)
    test = InverseTransform(trY_test, scalar1)
    
    #Calculate the error
    print "Finished Training..."
    return train, trainPredict, test, testPredict


#The main function
if __name__ == '__main__':
    '''
    if len(sys.argv) > 1: 
        DUMP_PATH = sys.argv[1]
    else: 
        DUMP_PATH = os.path.basename(__file__).split('.')[0]
    '''
    DUMP_PATH = "Results"
    if not os.path.exists(DUMP_PATH):
        os.makedirs(DUMP_PATH)
    print "Files are saved to: ", DUMP_PATH
    
    print "Data Type: ", DATA_TYPE
    print "Model Type: ", MODEL_TYPE
    # generate raw features
    print "Generating Features....."
    if DATA_TYPE == 'USStates': 
        data_reader = FeatureGeneration.DataReaderUSFlu()
    
    trX_raw, trY_raw, scalar1 = data_reader.GenerateNodeFeatures()
    # generate node classification 
    node_to_type = data_reader.GenerateNodeClassification()
    # generate graph
    if MODEL_TYPE == 'SRNNUndirected': 
        graph = data_reader.GenerateGraph()
        srnn_model = srnn_tf.SRNNUndirected
    elif MODEL_TYPE == 'SRNNDirected': 
        graph = data_reader.GenerateGraph()
        srnn_model = srnn_tf.SRNNDirected        
    
    # train the predictor.
    train, trainPredict, test, testPredict = RNNPrediction(trX_raw, trY_raw, scalar1, graph, node_to_type, srnn_model)
    
    if not os.path.exists(DUMP_PATH):
        os.makedirs(DUMP_PATH)
    for node in train:
        np.savetxt(os.path.join(DUMP_PATH, 'Train_N%d.csv' % node), train[node])
        np.savetxt(os.path.join(DUMP_PATH, 'TrainPredict_N%d.csv' % node), trainPredict[node])
        np.savetxt(os.path.join(DUMP_PATH, 'Test_N%d.csv' % node), test[node])
        np.savetxt(os.path.join(DUMP_PATH, 'TestPredict_N%d.csv' % node), testPredict[node])
    
    with open(os.path.join(DUMP_PATH, 'node_index.csv'), 'w') as f: 
        json.dump(data_reader.GenerateNodeIndex(), f)