'''
    Copyright 2017-2022 Department of Electrical and Computer Engineering
    University of Houston, TX/USA
    
    This file is part of UHVision Libraries.
    
    UH Vision libraries are free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    UH Vision Libraries are distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License along 
    with licenses for all the 3rd party libraries used in this repository. 
    If not, see <http://www.gnu.org/licenses/>. 
    Please contact Hien Nguyen V for more info about licensing hvnguy35@central.uh.edu, 
    and other members of the UHVision Lab via github issues section.
    **********************************************************************************
    Author:   Ilker GURCAN
    Date:     2/6/17
    File:     nn_model
    Comments: This file contains the architecture for NTM which is based
     on the memory-access model defined in the paper called
     "Meta-Learning with Memory-Augmented Neural Networks"
    **********************************************************************************
'''


import math
import tensorflow as tf


G_OUT = 'gOut_'
PARAM_MAP = 'paramMap_'


class ParamMap(object):

    __weights = None
    __biases = None

    def __init__(self, weights, biases):

        self.__weights = weights
        self.__biases = biases

    def get_weights(self):
        return self.__weights

    def get_biases(self):
        return self.__biases


class Memory(object):

class ModifiedLSTM(object):

    # Output by LSTM cell
    __cellContent = 0  # So called "State of the Cell" C_t
    __h_t = 0  # Output produced by the current Cell h_t
    # Config options
    __batch_size = 0
    __num_classes = 0
    __input_size = 0
    __controller_size = 0
    __memory_size = 0
    __num_read_heads = 0

    def __init__(self,
                 batch_size,
                 num_classes,
                 input_size,
                 controller_size,
                 memory_size,
                 num_read_heads):
        '''
        Constructor implementation for :class ModifiedLSTM

        :param batch_size: How many samples are processed at a given time t
        :param num_classes: How many classes are there to categorize our samples into
        :param input_size: Size of the input vector
        :param controller_size: Number of hidden units LSTM cell has
        :param memory_size: Shape of the memory (NumOfMemCells)X(LengthOfMemCell)
        :param num_read_heads: How many reading heads this LSTM cell is associated with
        '''

        self.__batch_size = batch_size
        self.__num_classes = num_classes
        self.__input_size = input_size
        self.__controller_size = controller_size
        self.__memory_size = memory_size
        self.__num_read_heads = num_read_heads

    def inference(self, x_t, cell_tm1=None, r_tm1=None, h_tm1=None):
        '''

        :param x_t:  Current input data -- (batch_size)x(num_classes+input_size)
        :param cell_tm1: Content of the previous cell at time t-1 -- (batch_size)x(controller_size)
        :param r_tm1: Previous output of the read head at time t-1 -- (batch_size)x(num_read_heads*num_mem_loc)
        :param h_tm1: Previous output of the hidden layer at time t-1 -- (batch_size)x(controller_size)
        :return:
        '''

        # Note that in this paper, they present previous output label as
        # input to the system; hence total number of inputs is equal to
        # "input_size + num_classes"
        num_inputs = self.__input_size+self.__num_classes
        # *************************************Hidden Layer*************************************
        with tf.variable_scope("hidden"):
            gate_params = {}  # Param tensors for each gate input-->forget-->output in order
            gate_out = {}  # Output tensors for each gate input-->forget-->output in order
            approx_cont = None  # Output tensor for approximated cell content
            approx_cont_params = []  # Param tensors for approximated cell content
            weights = []  # List of weights for each component
            for i in range(4):  # input-->forget-->output-->approximated cell content
                # Weights associating input to the current hidden units
                w_xh = tf.Variable(tf.truncated_normal([num_inputs, self.__controller_size],
                                                       stddev=1.0 / math.sqrt(float(num_inputs))),
                                   name="w_xh_"+str(i))
                # Weights associating previously read data from the memory to the current hidden unit
                num_tot_reads = self.__num_read_heads*self.__memory_size[1]  # Number of total read weights
                w_rh = tf.Variable(tf.truncated_normal([num_tot_reads,
                                                        self.__controller_size],
                                                       stddev=1.0 / math.sqrt(float(num_tot_reads))),
                                   name="w_rh_"+str(i))
                # Weights associating output of the previous hidden unit to the current hidden unit
                w_hh = tf.Variable(tf.truncated_normal([self.__controller_size, self.__controller_size],
                                                       stddev=1.0 / math.sqrt(float(self.__controller_size))),
                                   name="w_hh_"+str(i))
                # Single bias for each hidden unit
                biases = tf.Variable(tf.zeros([self.__controller_size]), name="biases_"+str(i))
                out = tf.nn.sigmoid(tf.matmul(x_t, w_xh) +
                                    tf.matmul(r_tm1, w_rh) +
                                    tf.matmul(h_tm1, w_hh) +
                                    biases, name="sig_"+str(i))  # (batch_size)x(controller_size)
                weights.append(w_xh)
                weights.append(w_rh)
                weights.append(w_hh)
                if i < 3:
                    gate_out[G_OUT+str(i)] = out
                    gate_params[PARAM_MAP+str(i)] = ParamMap(weights, biases)
                else:
                    approx_cont = out
                    approx_cont_params += weights
                weights.clear()
            # End of for-Loop
        # *************************************LSTM State*************************************
        with tf.variable_scope("out"):
            # State for the LSTM cell
            self.__cellContent = gate_out[G_OUT+str(1)]*cell_tm1+gate_out[G_OUT+str(0)]*approx_cont
            self.__h_t = gate_out[G_OUT+str(i)]*tf.nn.tanh(self.__cellContent)
        # *************************************Controller Output Layer*************************************
        with tf.variable_scope('cOut'):
            w_key = tf.Variable(tf.truncated_normal([self.__controller_size,
                                                     self.__memory_size[1],
                                                     self.__num_read_heads],
                                                    stddev=1.0 / math.sqrt(float(self.__controller_size))),
                                name="w_key")
            b_key = tf.Variable(tf.zeros([self.__controller_size]), name="bias_key")
            w_add = tf.Variable(tf.truncated_normal([self.__controller_size,
                                                     self.__memory_size[1],
                                                     self.__num_read_heads],
                                                    stddev=1.0 / math.sqrt(float(self.__controller_size))),
                                  name="w_add")
            b_add = tf.Variable(tf.zeros([self.__controller_size]), name="bias_add")
            w_sig = tf.Variable(tf.truncated_normal([self.__controller_size,
                                                     1,
                                                     self.__num_read_heads],
                                                    stddev=1.0 / math.sqrt(float(self.__controller_size))),
                                  name="w_sig")
            b_sig = tf.Variable(tf.zeros([self.__controller_size]), name="bias_sig")
            # Calculating key, add, alpha (learnable sigmoid gate parameter)
            k_t = tf.nn.tanh(tf.matmul(self.__h_t, w_key) + b_key)
            a_t = tf.nn.tanh(tf.matmul(self.__h_t, w_add) + b_add)
            s_t = tf.nn.tanh(tf.matmul(self.__h_t, w_sig) + b_sig)

        return gate_out, gate_params, approx_cont, approx_cont_params


'''
    -Defines the network + memory models.
    -Each controller unit is a modified LSTM cell. By "modified" it is denoted that
    an extra input which is the previously accessed data from the memory by read head,
    passed to the LSTM cell along with x_t, current input to the system & h_(t-1) output
    from previous step.
    -
'''


class NNModel(object):

    __controller_size = 0  # How
    __num_classes = 0  # Number of classes/labels
    __memory_size = (0, 0)  # 2D Tuple specifying (numberOfElements)X(sizeOfEachElement)
    __num_read_heads = 0  # Number of read heads attached to controller

    def __init__(self,
                 controller_size,
                 num_classes,
                 memory_size,
                 num_read_heads):
        self.__controller_size = controller_size
        self.__num_classes = num_classes
        self.__memory_size = memory_size
        self.__num_read_heads = num_read_heads

