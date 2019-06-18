"""
Tensorflow implementation of bi-directional SRNN.

Author: Bao Wang, Xiyang Luo, Fangbo Zhang
    Department of Mathematics, UCLA
Email: wangbaonj@gmail.com
       xylmath@gmail.com
       fb.zhangsjtu@gmail.com
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import copy
import os

DECAY_STEP = 40
DECAY_RATE = 0.4


def _default_node_cell(): 
    # define the lstm cell for node rnn
    num_units = 20
    return tf.contrib.rnn.BasicLSTMCell(num_units=num_units)
    

def _default_edge_cell(): 
    # define lstm cell(s) for edge rnn
    num_units = [10,20,20,10]
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.BasicLSTMCell(num_units=nu) for nu in num_units])
    return stacked_lstm


def TrainValSplit(trX, trY, ratio=0.9, shuffle=False): 
    """ Helper for train validation split given trX, trY.
    """
    N = trY[trY.keys()[0]].shape[0]
    num_train = int(N * ratio)
    if shuffle:
        ind = np.random.permutation(N)
    else:
        ind = np.arange(N).astype(np.int)
    ind_train = ind[:num_train]
    ind_val = ind[num_train:]
    trX_train = {}
    trX_val = {}
    trY_train = {}
    trY_val = {}
    for node in trX:
        trX_train[node] = {}
        trY_train[node] = trY[node][ind_train, :]
        trX_val[node] = {}
        trY_val[node] = trY[node][ind_val, :]
        for e in trX[node]: 
            trX_train[node][e] = trX[node][e][ind_train, :, :]
            trX_val[node][e] = trX[node][e][ind_val, :, :]

    return trX_train, trX_val, trY_train, trY_val


def InverseTransform(trY, scalar1):
    # inverse transform of scaler applied node-wise.
    trY_uscaled = {}
    for node in trY:
        sp = trY[node].shape
        trY_uscaled[node] = np.reshape(scalar1.inverse_transform(trY[node].reshape(-1, 1)).flatten(), newshape=sp)
    return trY_uscaled


class SRNNBase(): 
    """ Tensorflow implementation of undirected srnn 
    Args: 
        graph: the node-edge representation of graph. dictionary of {node(int): [node1(int), node2(int)...]}
        node_to_type: dictionary from node_id(int) to class_id(int). dictionary of {node: type(or class)}
        node_input_dims: input feature dimension. D in the (N, T, D) 3D tensor. 
        output_dims: output label dimension. 
        seq_length: length of sequence. T in the (N, T, D) 3D tensor. 
        node_cell: a function handle that generates cells used for node_rnn.
        edge_cell: a function handle that generates cells used for edge_rnn.
        use_dropout: a boolean to indicate whether to use dropout. 
        dropout_rate: dropout rate. 
    """
    def __init__(
        self, graph, node_to_type, node_input_dims=1, 
        edge_input_dims=1, output_dims=1, seq_length=None, 
        use_dropout=False, dropout_rate=0.2):  
        self.node_input_dims = node_input_dims
        self.edge_input_dims = edge_input_dims
        self.output_dims = output_dims
        self.seq_length = seq_length
        self.node_graph = graph
        self.nodes = graph.keys()
        self.node_to_type = node_to_type
        self.types = list(set(node_to_type.values()))
        self.type_to_edge_connections = {}
        self.use_dropout = use_dropout
        if use_dropout: 
            self.dropout_rate = dropout_rate
        
    def build_model(self): 
        """Function to construct model graph. """
        node_rnn_outputs = {}
        edge_rnn_outputs = {}
        placeholders = {}
        if self.use_dropout: 
            is_train = tf.placeholder(tf.bool)
            placeholders['is_train'] = is_train
        for t in self.types: 
            edges = self.type_to_edge_connections[t]
            edge_to_current_node = []
            # construct outputs from all edges rnns connected to t.
            for edge in edges: 
                # construct placeholder for edgernn node_input
                if edge.split('_')[-1] == 'input':  
                    with tf.variable_scope('Edge_' + edge): 
                        X = tf.placeholder(dtype=tf.float32, shape=[None, self.seq_length, self.node_input_dims])
                    placeholders[edge] = X
                # construct placeholder for edgernn node_node
                elif edge not in placeholders:
                    with tf.variable_scope('Edge_' + edge): 
                        X = tf.placeholder(dtype=tf.float32, shape=[None, self.seq_length, self.edge_input_dims])
                    placeholders[edge] = X
                # construct edge rnn output if not already constructed
                if edge not in edge_rnn_outputs: 
                    X = placeholders[edge]
                    cell = self.edge_cells[edge]
                    with tf.variable_scope('Edge_' + edge): 
                        edge_outputs, _ = tf.nn.dynamic_rnn(  # (N, T, D)
                            cell=cell,
                            dtype=tf.float32,
                            inputs=X)
                        edge_rnn_outputs[edge] = edge_outputs
                else:
                    edge_outputs = edge_rnn_outputs[edge]
                edge_to_current_node += [edge_outputs]
            # concatenate outputs from edgernn and feed to nodernn
            node_rnn_inputs = tf.concat(edge_to_current_node, axis=2)
            if self.use_dropout: 
                node_rnn_inputs = tf.layers.dropout(node_rnn_inputs, rate=self.dropout_rate, training=is_train)
        
            cell = self.node_cells[t]
            # construct node_rnn output
            with tf.variable_scope('Node_' + str(t)): 
                node_outputs, _ = tf.nn.dynamic_rnn(  # (N, T, D)
                    cell=cell,
                    dtype=tf.float32,
                    inputs=node_rnn_inputs)
            # fully connected layer to output dimension
            if self.use_dropout: 
                node_outputs = tf.layers.dropout(node_outputs, rate=self.dropout_rate, training=is_train)
            output_activation = tf.tanh
            final_output = tf.layers.dense(
                inputs=node_outputs[:, -1, :], units=self.output_dims, activation=output_activation)
            node_rnn_outputs[t] = final_output
        # construct output_placeholders
        output_placeholders = {}
        for t in self.types: 
            output_placeholders[t] = tf.placeholder(dtype=tf.float32, shape=[None, self.output_dims])
        loss_per_type = {}
        # compute loss tensor for each node_type. 
        for t in self.types: 
            Y_pr = node_rnn_outputs[t]
            Y = output_placeholders[t]
            loss_per_type[t] = tf.reduce_mean(tf.square(Y - Y_pr))

            # record the outputs
            self.loss_per_type = loss_per_type
            self.output_placeholders = output_placeholders
            self.edge_rnn_outputs = edge_rnn_outputs
            self.node_rnn_outputs = node_rnn_outputs
            self.placeholders = placeholders

    def _shuffle_data(self, trX, trY, N): 
        ind = np.random.permutation(N)
        for node in trX: 
            edges = self.type_to_edge_connections[self.node_to_type[node]]
            for edge in edges: 
                trX[node][edge] = trX[node][edge][ind, :, :]
            trY[node] = trY[node][ind, :]
        return trX, trY
           

    def load_model(self, sess, model_path, global_step=None):
        """ Helper function to load saved models. """
        with open(os.path.join(model_path, "checkpoint"), "r") as f:
            name = f.readline()
        model_name, model_id = name.split('"')[1].split('-')
        if not global_step: 
            model_name = model_name + '-' + model_id
        else:
            model_name = model_name + '-' + str(global_step)
        saver = tf.train.import_meta_graph(os.path.join(model_path, model_name+'.meta'))
        saver.restore(sess, os.path.join(model_path, model_name))


    def fit_model(
            self, sess, trX, trY, max_iter=150, learning_rate=0.01, batch_size=128, validation_batch_size=50, 
            shuffle_examples=True,  num_epochs_per_summary=1, validation_split=0.2,
            save_model_epoch=-1, save_model_dir='.', start_from_checkpoint=False, checkpoint_dir='.', 
            sample_rate=1.0): 
        """Function to fit model given formatted training data. 

        Args: 
            sess: tensorflow session object. 
            trX: training features. A dictionary of form {node: {edge_name: X}}, 
                where X is a 3D tensor of shape (N, T, D), N being number of examples, 
                T the sequence length and D the input dimensions. 
            trY: training labels. A dictionary of form {node: Y}, where Y is a 
                2D tensor of shape (N, D). 
            max_iter: max epochs to train for. 
            shuffle_examples: whether to shuffle examples at start of each epoch. 
            num_epochs_per_summary: number of epochs to output summary. 
            validation_split: fraction of data used to compute validation loss. 
            fit_per_type: fit node/edge RNN for every type or every node. (Default per Type, do not change). 
            save_model_epoch: epochs to save model. 
            save_model_dir: directory to save model. 
            start_from_checkpoint: Flag to indicate whether to train from scratch or load existing checkpoints. 
            checkpoint_dir: initial model checkpoint directory.

        """
        # deep copy training data to avoid shuffling
        temp1 = copy.deepcopy(trX)
        temp2 = copy.deepcopy(trY)
        N = trY[trY.keys()[0]].shape[0]
        trX = temp1
        trY = temp2
        # train validation split
        if validation_split > 1e-6:
            num_train = int(N * (1.0 - validation_split))
            ind = np.random.permutation(N)
            ind_train = ind[:num_train]
            ind_val = ind[num_train:]
            trX_train = {}
            trX_val = {}
            trY_train = {}
            trY_val = {}
            for node in trX:
                trX_train[node] = {}
                trY_train[node] = trY[node][ind_train, :]
                trX_val[node] = {}
                trY_val[node] = trY[node][ind_val, :]
                for e in trX[node]: 
                    trX_train[node][e] = trX[node][e][ind_train, :, :]
                    trX_val[node][e] = trX[node][e][ind_val, :, :]
            trX = trX_train 
            trY = trY_train 
            N = trY[trY.keys()[0]].shape[0]

        # use the ADAM optimizer by default
        beta1 = tf.Variable(0.9, name='beta1')
        beta2 = tf.Variable(0.999, name='beta2') 
        # add exponential learning rate decay in training. 
        initial_learning_rate = learning_rate
        print "Initial learning rate: ", initial_learning_rate
        global_step_placeholder = tf.placeholder(tf.int32)
        learning_rate = tf.train.exponential_decay(
            learning_rate, global_step_placeholder, DECAY_STEP, DECAY_RATE)
        print "Decay Step: ", DECAY_STEP
        print "Decay Rate: ", DECAY_RATE

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
        

        output_placeholders = self.output_placeholders
        loss_per_type = self.loss_per_type
        # compute train op each node_type. 
        train_op_per_type = {}
        for t in self.types: 
            
            #tv=tf.trainable_variables()
            
            #l1
            #weights=[v for v in tv if len(v.shape)>1]
            #l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.000005, scope=None)
            #regularization_cost = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
            
            #l2:
            #regularization_cost=0.0005*tf.reduce_sum([tf.nn.l2_loss(v) for v in tv if len(v.shape)>1])
            
            #l0: NOT WORKING
            #regularization_cost=0.00005*tf.sum_n([tf.count_nonzero(v) for v in tv if len(v,shape)>1])
            
            
            #train_op_per_type[t] = optimizer.minimize(loss_per_type[t]+regularization_cost)
            
            
            #no regularization
            train_op_per_type[t] = optimizer.minimize(loss_per_type[t])
            
            
            #l0 thresholding NOT WORKING
            #tv=tf.trainable_variables()
            #for v in tv:
                #if len(v.shape)>1:
                    #mask=tf.zeros(v.shape)
                    #cond=tf.less_equal(v,tf.constant(0.02))
                    #v=tf.where(cond,v,mask)

        if save_model_epoch > 0:
            saver = tf.train.Saver()

        if start_from_checkpoint:
            self.load_model(sess, checkpoint_dir)
        else:
            sess.run(tf.global_variables_initializer())   
        # start training
        num_steps_per_epoch = int(np.floor(N / batch_size))
        # num_steps_per_epoch = 2
        print "Start Training......"
        print "batch_size = ", batch_size
        print "num_steps_per_epoch", num_steps_per_epoch
        print "sample_rate : ", sample_rate
        for epoch in xrange(max_iter): 
            if shuffle_examples: 
                trX, trY = self._shuffle_data(trX, trY, N)
            total_loss_per_epoch = 0
            t0 = time.time()
            tprev = float(time.time())
            # randomly sample nodes to train on if needed.
            node_list = trX.keys()
            if sample_rate < 1.0: 
                np.random.shuffle(node_list)
                num_nodes_after_sample = int(len(node_list) * sample_rate)
                node_list = node_list[:num_nodes_after_sample]
            for step in xrange(num_steps_per_epoch+1): 
                # for each type, run one step of optimizer on minimatch
                node_types = self.types
                for t in node_types: 
                    train_labels = np.zeros(shape=[0, self.output_dims])
                    train_op = train_op_per_type[t]
                    loss = loss_per_type[t]
                    train_vals = {}
                    edges = self.type_to_edge_connections[t]
                    ind_start = step * batch_size
                    ind_end = (step + 1) * batch_size
                    # concatenate all data from node with the same type. 
                    for edge in edges: 
                        if edge.split('_')[-1] == 'input':
                            D = self.node_input_dims
                        else: 
                            D = self.edge_input_dims
                        train_vals[edge] = np.zeros(shape=[0, self.seq_length, D])
                    for node in trX.keys(): 
                        if node in node_list and self.node_to_type[node] == t: 
                            for edge in edges: 
                                train_vals[edge] = np.concatenate(
                                    [train_vals[edge], trX[node][edge][ind_start:ind_end, :, :]], axis=0)
                            train_labels = np.concatenate([train_labels, trY[node][ind_start:ind_end, :]], axis=0)
                    # construct the feed_dict for the train_op
                    feed_dict = {}
                    if self.use_dropout: 
                        feed_dict[self.placeholders['is_train']] = True
                    for edge in edges: 
                        placeholder_var = self.placeholders[edge]
                        feed_dict[placeholder_var] = train_vals[edge]
                    feed_dict[output_placeholders[t]] = train_labels
                    feed_dict[global_step_placeholder] = epoch
                    # run training and get loss. 
                    lv, _ = sess.run([loss, train_op], feed_dict=feed_dict)   
                    total_loss_per_epoch += lv

            total_loss_per_epoch /= num_steps_per_epoch
            if epoch % num_epochs_per_summary == 0: 
                if validation_split > 1e-6:
                    validation_loss = self.compute_loss(sess, trX_val, trY_val, validation_batch_size)
                t1 = time.time()
                time_per_epoch = float(t1 - t0) / num_epochs_per_summary
                if validation_split > 1e-6:
                    print "Epoch: %d Train Loss: %.6f Val Loss %.6f Time per Epoch: %.2fs" % (epoch, total_loss_per_epoch, validation_loss, time_per_epoch)
                else:
                    print "Epoch: %d Train Loss: %.6f Time per Epoch: %.2fs" % (epoch, total_loss_per_epoch, time_per_epoch)
                t0 = t1
            if save_model_epoch > 0:
                if epoch % save_model_epoch == 0 and epoch >= 1:
                    saver.save(sess, os.path.join(save_model_dir, 'srnnmodel'), global_step=epoch)
        #tf.train.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt')
        variables_names =[v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
        '''
        with open('Exported.txt', 'w') as file:
            for k,v in zip(variables_names, values):
                print k
                print v.shape
                file.write(k)
                file.write('\n')
                file.write(str(v.tolist()))
                file.write('\n')
        '''
        fig=plt.figure(figsize=(10,50))
        i=1
        for k,v in zip(variables_names, values):
            if len(v.shape)>1:
                print k
                print v.shape
                print tf.count_nonzero(v).eval()
                ax=fig.add_subplot(12,2,2*i-1)
                ax.hist(v.reshape(v.shape[0]*v.shape[1],))
                ax=fig.add_subplot(12,2,2*i)
                ax.hist(v.reshape(v.shape[0]*v.shape[1],),range=(-0.05,0.05))
                i=i+1
        #fig.savefig('hist.png')    
           
    
    def predict_node(self, sess, trX): 
        """Function to fit model given formatted training data. 

        Args: 
            sess: tensorflow session object. 
            trX: training features. A dictionary of form {node: {edge_name: X}}, 
                where X is a 3D tensor of shape (N, T, D), N being number of examples, 
                T the sequence length and D the input dimensions. 
        Returns: 
            trY: training labels. A dictionary of form {node: Y}, where Y is a 
                2D tensor of shape (N, D). 
        """
        trY = {}
        for node in trX: 
            t = self.node_to_type[node]
            feed_dict={}
            if self.use_dropout: 
                feed_dict[self.placeholders['is_train']] = False
            edges = self.type_to_edge_connections[t]
            for edge in edges: 
                feed_dict[self.placeholders[edge]] = trX[node][edge]
            trY[node] = sess.run(self.node_rnn_outputs[t], feed_dict=feed_dict)
        return trY

    def compute_loss(self, sess, trX, trY, validation_batch_size=64): 
        """Function to fit model given formatted training data. 

        Args: 
            sess: tensorflow session object. 
            trX: training features. A dictionary of form {node: {edge_name: X}}, 
                where X is a 3D tensor of shape (N, T, D), N being number of examples, 
                T the sequence length and D the input dimensions. 
        Returns: 
            trY: training labels. A dictionary of form {node: Y}, where Y is a 
                2D tensor of shape (N, D). 
        """
        loss = 0
        node_types = self.types
        output_placeholders = self.output_placeholders
        N = trY[trY.keys()[0]].shape[0]
        if validation_batch_size > 0:
            batch_size = validation_batch_size
            num_steps_per_epoch = int(np.floor(N / batch_size))
        else:
            batch_size = N
            num_steps_per_epoch = 1
        for step in xrange(num_steps_per_epoch): 
            # for each type, run one step of optimizer on minimatch
            node_types = self.types
            for t in node_types: 
                edges = self.type_to_edge_connections[t]
                train_labels = np.zeros(shape=[0, self.output_dims])
                train_vals = {}
                edges = self.type_to_edge_connections[t]
                ind_start = step * batch_size
                ind_end = (step + 1) * batch_size
                # concatenate all data from node with the same type. 
                for edge in edges: 
                    if edge.split('_')[-1] == 'input':
                        D = self.node_input_dims
                    else: 
                        D = self.edge_input_dims
                    train_vals[edge] = np.zeros(shape=[0, self.seq_length, D])
                for node in trX.keys(): 
                    if self.node_to_type[node] == t: 
                        for edge in edges: 
                            train_vals[edge] = np.concatenate(
                                [train_vals[edge], trX[node][edge][ind_start:ind_end, :, :]], axis=0)
                        train_labels = np.concatenate([train_labels, trY[node][ind_start:ind_end, :]], axis=0)
                # construct the feed_dict for the train_op
                feed_dict = {}
                if self.use_dropout: 
                    feed_dict[self.placeholders['is_train']] = False
                for edge in edges: 
                    placeholder_var = self.placeholders[edge]
                    feed_dict[placeholder_var] = train_vals[edge]
                feed_dict[output_placeholders[t]] = train_labels
                # run training and get loss. 
                loss += sess.run(self.loss_per_type[t], feed_dict=feed_dict) 
        loss /= num_steps_per_epoch
        return loss

    def add_edge_features(self, trX_raw, trY_raw, external_dims=5): 
        raise NotImplementedError('Not implemented')
            

class SRNNUndirected(SRNNBase): 
    """ Tensorflow implementation of undirected srnn 

    Args: 
        graph: the node-edge representation of graph. dictionary of {node(int): [node1(int), node2(int)...]}
        node_to_type: dictionary from node_id(int) to class_id(int). dictionary of {node: type(or class)}
        node_input_dims: input feature dimension. D in the (N, T, D) 3D tensor. 
        output_dims: output label dimension. 
        seq_length: length of sequence. T in the (N, T, D) 3D tensor. 
        node_cell: a function handle that generates cells used for node_rnn.
        edge_cell: a function handle that generates cells used for edge_rnn.
        use_dropout: a boolean to indicate whether to use dropout. 
        dropout_rate: dropout rate. 
    """
    def __init__(
        self, graph, node_to_type, node_input_dims=1, 
        edge_input_dims=1, output_dims=1, seq_length=None, 
        node_cell=None, edge_cell=None, use_dropout=False, dropout_rate=0.2): 

        # call the base initializer
        SRNNBase.__init__(
            self, graph, node_to_type, node_input_dims=node_input_dims, 
            edge_input_dims=edge_input_dims, output_dims=output_dims, seq_length=seq_length, 
            use_dropout=use_dropout, dropout_rate=dropout_rate)

        if node_cell is None: 
            node_cell = _default_node_cell
        if edge_cell is None: 
            edge_cell = _default_edge_cell 
        # construct undirected type graph
        for t in self.types:
            self.type_to_edge_connections[t] = []
        type_edges = set()
        for node in graph: 
            t = node_to_type[node]
            ms = graph[node]
            for m in ms: 
                t2 = node_to_type[m]
                ind1 = min(t, t2)
                ind2 = max(t, t2)
                tok = str(ind1) + '_' + str(ind2)
                if(tok not in type_edges):
                    type_edges.add(tok)
                    if tok not in self.type_to_edge_connections[t]: 
                        self.type_to_edge_connections[t] += [tok]
                    if tok not in self.type_to_edge_connections[t2]: 
                        self.type_to_edge_connections[t2] += [tok]

        # add in the input as an edgernn
        for t in self.types:
            tok = str(t) + '_' + 'input'
            type_edges.add(tok)
            self.type_to_edge_connections[t] += [tok]

        # construct lstm cells for nodeRNN and edgeRNN
        edge_cells = {}
        for tok in type_edges: 
            with tf.variable_scope('Edge_' + tok): 
                cell = edge_cell()  # edge rnn cells. 
                edge_cells[tok] = cell
        self.edge_cells = edge_cells
        node_cells = {}
        for t in self.types: 
            with tf.variable_scope('Node_' + str(t)): 
                cell = node_cell()  # node rnn cells.
                node_cells[t] = cell    
        self.node_cells = node_cells
        
    def build_model(self): 
        """Function to construct model graph. """
        node_rnn_outputs = {}
        edge_rnn_outputs = {}
        placeholders = {}
        if self.use_dropout: 
            is_train = tf.placeholder(tf.bool)
            placeholders['is_train'] = is_train
        for t in self.types: 
            edges = self.type_to_edge_connections[t]
            edge_to_current_node = []
            # construct outputs from all edges rnns connected to t.
            for edge in edges: 
                # construct placeholder for edgernn node_input
                if edge.split('_')[-1] == 'input':  
                    with tf.variable_scope('Edge_' + edge): 
                        X = tf.placeholder(dtype=tf.float32, shape=[None, self.seq_length, self.node_input_dims])
                    placeholders[edge] = X
                # construct placeholder for edgernn node_node
                elif edge not in placeholders:
                    with tf.variable_scope('Edge_' + edge): 
                        X = tf.placeholder(dtype=tf.float32, shape=[None, self.seq_length, self.edge_input_dims])
                    placeholders[edge] = X
                # construct edge rnn output if not already constructed
                if edge not in edge_rnn_outputs: 
                    X = placeholders[edge]
                    cell = self.edge_cells[edge]
                    with tf.variable_scope('Edge_' + edge): 
                        edge_outputs, _ = tf.nn.dynamic_rnn(  # (N, T, D)
                            cell=cell,
                            dtype=tf.float32,
                            inputs=X)
                        edge_rnn_outputs[edge] = edge_outputs
                else:
                    edge_outputs = edge_rnn_outputs[edge]
                edge_to_current_node += [edge_outputs]
            
            #Xiyang:  If you want Dense from the edge output here, use tf.layers.dense.    
            # concatenate outputs from edgernn and feed to nodernn
            node_rnn_inputs = tf.concat(edge_to_current_node, axis=2)
            if self.use_dropout: 
                node_rnn_inputs = tf.layers.dropout(node_rnn_inputs, rate=self.dropout_rate, training=is_train)
        
            cell = self.node_cells[t]
            # construct node_rnn output
            with tf.variable_scope('Node_' + str(t)): 
                node_outputs, _ = tf.nn.dynamic_rnn(  # (N, T, D)
                    cell=cell,
                    dtype=tf.float32,
                    inputs=node_rnn_inputs)
            # fully connected layer to output dimension
            if self.use_dropout: 
                node_outputs = tf.layers.dropout(node_outputs, rate=self.dropout_rate, training=is_train)
            output_activation = tf.tanh
            final_output = tf.layers.dense(
                inputs=node_outputs[:, -1, :], units=self.output_dims, activation=output_activation)
            node_rnn_outputs[t] = final_output
        # construct output_placeholders
        output_placeholders = {}
        for t in self.types: 
            output_placeholders[t] = tf.placeholder(dtype=tf.float32, shape=[None, self.output_dims])
        loss_per_type = {}
        # compute loss tensor for each node_type. 
        for t in self.types: 
            Y_pr = node_rnn_outputs[t]
            Y = output_placeholders[t]
            loss_per_type[t] = tf.reduce_mean(tf.square(Y - Y_pr))

            # record the outputs
            self.loss_per_type = loss_per_type
            self.output_placeholders = output_placeholders
            self.edge_rnn_outputs = edge_rnn_outputs
            self.node_rnn_outputs = node_rnn_outputs
            self.placeholders = placeholders

    def add_edge_features(self, trX_raw, trY_raw, external_dims=5): 
        """Helper function to construct edge features for training via weighted average pooling. 

        Args: 
            trX_raw: raw training features. A dictionary of form {node: X}, 
                where X is a 3D tensor of shape (N, T, D), N being number of examples, 
                T the sequence length and D the input dimensions. 
            trY_raw: training labels. A dictionary of form {node: Y}, where Y is a 
                2D tensor of shape (N, D). 
        Returns: 
            trX: training features. A dictionary of form {node: {edge_name: X}}, 
                where X is a 3D tensor of shape (N, T, D), N being number of examples, 
                T the sequence length and D the input dimensions. 
            trY: training labels. A dictionary of form {node: Y}, where Y is a 
                2D tensor of shape (N, D). 
        """
        trX = {}  # {node: {edge_name: input_value}}
        trY = trY_raw  # {node: label_value}
        graph = self.node_graph 
        type_to_edge_connections = self.type_to_edge_connections
        node_to_type = self.node_to_type
        for node in trX_raw: 
            trX[node] = {}
        for node in trX_raw: 
            t = node_to_type[node]
            N, T, D = trX_raw[node].shape
            edges = type_to_edge_connections[t]
            edge_input_values = {}
            edge_counts = {}
            edge_input_values[str(t) + '_' + 'input'] = trX_raw[node]
            # for now, just add the features from neighbors of the same
            # type to form edge_features. One can also add in correlation
            # ... later. 
            for edge in edges: 
                if edge.split('_')[-1] != 'input': 
                    edge_input_values[edge] = np.zeros(shape=[N, T, D])
                    edge_input_values[edge][:, :external_dims, :] = trX_raw[node][:, :external_dims, :]
            nbs = graph[node]
            for nb in nbs:
                if type(nb) is dict: 
                    weights = nbs[nb]
                else:
                    weights = 1.0
                t2 = node_to_type[nb]
                ind1 = min(t, t2)
                ind2 = max(t, t2)
                tok = str(ind1) + '_' + str(ind2)
                edge_input_values[tok][:, external_dims:, :] += (trX_raw[nb][:, external_dims:, :] * weights)
            trX[node] = edge_input_values           
        return trX, trY


class SRNNDirected(SRNNBase): 
    """ Tensorflow implementation of directed srnn 

    Args: 
        graph: the node-edge representation of graph. dictionary of {node(int): [node1(int), node2(int)...]}. 
            represents edge from node1<---node2.
        node_to_type: dictionary from node_id(int) to class_id(int). dictionary of {node: type(or class)}
        node_input_dims: input feature dimension. D in the (N, T, D) 3D tensor. 
        output_dims: output label dimension. 
        seq_length: length of sequence. T in the (N, T, D) 3D tensor. 
        node_cell: a function handle that generates cells used for node_rnn.
        edge_cell: a function handle that generates cells used for edge_rnn.
        use_dropout: a boolean to indicate whether to use dropout. 
        dropout_rate: dropout rate. 
    """
    def __init__(
        self, graph, node_to_type, node_input_dims=1, 
        edge_input_dims=1, output_dims=1, seq_length=None, 
        node_cell=None, edge_cell=None, use_dropout=False, dropout_rate=0.2): 

        # call the base initializer
        SRNNBase.__init__(
            self, graph, node_to_type, node_input_dims=1, 
            edge_input_dims=edge_input_dims, output_dims=output_dims, seq_length=seq_length, 
            use_dropout=use_dropout, dropout_rate=dropout_rate)

        if node_cell is None: 
            node_cell = _default_node_cell
        if edge_cell is None: 
            edge_cell = _default_edge_cell 
        # construct undirected type graph
        for t in self.types:
            self.type_to_edge_connections[t] = []
        type_edges = set()
        for node in graph: 
            ind1 = node_to_type[node]
            ms = graph[node]
            for m in ms: 
                ind2 = node_to_type[m]
                tok = str(ind1) + '_' + str(ind2)
                if(tok not in type_edges):
                    type_edges.add(tok)
                    if tok not in self.type_to_edge_connections[ind1]: 
                        self.type_to_edge_connections[ind1] += [tok]
                tok = str(ind2) + '_' + str(ind1)
                if(tok not in type_edges):
                    type_edges.add(tok)
                    if tok not in self.type_to_edge_connections[ind2]: 
                        self.type_to_edge_connections[ind2] += [tok]

        # add in the input as an edgernn
        for t in self.types:
            tok = str(t) + '_' + 'input'
            type_edges.add(tok)
            self.type_to_edge_connections[t] += [tok]

        # construct lstm cells for nodeRNN and edgeRNN
        edge_cells = {}
        for tok in type_edges: 
            with tf.variable_scope('Edge_' + tok): 
                cell = edge_cell()  # edge rnn cells. 
                edge_cells[tok] = cell
        self.edge_cells = edge_cells
        node_cells = {}
        for t in self.types: 
            with tf.variable_scope('Node_' + str(t)): 
                cell = node_cell()  # node rnn cells.
                node_cells[t] = cell    
        self.node_cells = node_cells
        
    def build_model(self): 
        """Function to construct model graph. """
        node_rnn_outputs = {}
        edge_rnn_outputs = {}
        placeholders = {}
        if self.use_dropout: 
            is_train = tf.placeholder(tf.bool)
            placeholders['is_train'] = is_train
        for t in self.types: 
            edges = self.type_to_edge_connections[t]
            edge_to_current_node = []
            # construct outputs from all edges rnns connected to t.
            for edge in edges: 
                # construct placeholder for edgernn node_input
                if edge.split('_')[-1] == 'input':  
                    with tf.variable_scope('Edge_' + edge): 
                        X = tf.placeholder(dtype=tf.float32, shape=[None, self.seq_length, self.node_input_dims])
                    placeholders[edge] = X
                # construct placeholder for edgernn node_node
                elif edge not in placeholders:
                    with tf.variable_scope('Edge_' + edge): 
                        X = tf.placeholder(dtype=tf.float32, shape=[None, self.seq_length, self.edge_input_dims])
                    placeholders[edge] = X
                # construct edge rnn output if not already constructed
                if edge not in edge_rnn_outputs: 
                    X = placeholders[edge]
                    cell = self.edge_cells[edge]
                    with tf.variable_scope('Edge_' + edge): 
                        edge_outputs, _ = tf.nn.dynamic_rnn(  # (N, T, D)
                            cell=cell,
                            dtype=tf.float32,
                            inputs=X)
                        edge_rnn_outputs[edge] = edge_outputs
                else:
                    edge_outputs = edge_rnn_outputs[edge]
                edge_to_current_node += [edge_outputs]
            # concatenate outputs from edgernn and feed to nodernn
            node_rnn_inputs = tf.concat(edge_to_current_node, axis=2)
            if self.use_dropout: 
                node_rnn_inputs = tf.layers.dropout(node_rnn_inputs, rate=self.dropout_rate, training=is_train)
        
            cell = self.node_cells[t]
            # construct node_rnn output
            with tf.variable_scope('Node_' + str(t)): 
                node_outputs, _ = tf.nn.dynamic_rnn(  # (N, T, D)
                    cell=cell,
                    dtype=tf.float32,
                    inputs=node_rnn_inputs)
            # fully connected layer to output dimension
            if self.use_dropout: 
                node_outputs = tf.layers.dropout(node_outputs, rate=self.dropout_rate, training=is_train)
            output_activation = tf.sigmoid
            final_output = tf.layers.dense(
                inputs=node_outputs[:, -1, :], units=self.output_dims, activation=output_activation)
            node_rnn_outputs[t] = final_output
        # construct output_placeholders
        output_placeholders = {}
        for t in self.types: 
            output_placeholders[t] = tf.placeholder(dtype=tf.float32, shape=[None, self.output_dims])
        loss_per_type = {}
        # compute loss tensor for each node_type. 
        for t in self.types: 
            Y_pr = node_rnn_outputs[t]
            Y = output_placeholders[t]
            loss_per_type[t] = tf.reduce_mean(tf.square(Y - Y_pr))

            # record the outputs
            self.loss_per_type = loss_per_type
            self.output_placeholders = output_placeholders
            self.edge_rnn_outputs = edge_rnn_outputs
            self.node_rnn_outputs = node_rnn_outputs
            self.placeholders = placeholders

    def add_edge_features(self, trX_raw, trY_raw, external_dims=5): 
        """Helper function to construct edge features for training. 

        Args: 
            trX_raw: raw training features. A dictionary of form {node: X}, 
                where X is a 3D tensor of shape (N, T, D), N being number of examples, 
                T the sequence length and D the input dimensions. 
            trY_raw: training labels. A dictionary of form {node: Y}, where Y is a 
                2D tensor of shape (N, D). 
        Returns: 
            trX: training features. A dictionary of form {node: {edge_name: X}}, 
                where X is a 3D tensor of shape (N, T, D), N being number of examples, 
                T the sequence length and D the input dimensions. 
            trY: training labels. A dictionary of form {node: Y}, where Y is a 
                2D tensor of shape (N, D). 
        """
        trX = {}  # {node: {edge_name: input_value}}
        trY = trY_raw  # {node: label_value}
        graph = self.node_graph 
        type_to_edge_connections = self.type_to_edge_connections
        node_to_type = self.node_to_type
        for node in trX_raw: 
            trX[node] = {}
        for node in trX_raw: 
            t = node_to_type[node]
            N, T, D = trX_raw[node].shape
            edges = type_to_edge_connections[t]
            edge_input_values = {}
            edge_counts = {}
            edge_input_values[str(t) + '_' + 'input'] = trX_raw[node]
            # for now, just add the features from neighbors of the same
            # type to form edge_features. One can also add in correlation
            # ... later. 
            for edge in edges: 
                if edge.split('_')[-1] != 'input': 
                    edge_input_values[edge] = np.zeros(shape=[N, T, D])
                    edge_input_values[edge][:, :external_dims, :] = trX_raw[node][:, :external_dims, :]
            nbs = graph[node]
            for nb in nbs:
                if type(nb) is dict: 
                    weights = nbs[nb]
                else:
                    weights = 1.0
                ind2 = node_to_type[nb]
                ind1 = t
                tok = str(ind1) + '_' + str(ind2)
                edge_input_values[tok][:, external_dims:, :] += (trX_raw[nb][:, external_dims:, :] * weights)
            trX[node] = edge_input_values           
        return trX, trY