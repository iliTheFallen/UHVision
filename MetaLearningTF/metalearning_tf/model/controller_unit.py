'''
    Copyright 2017-2022 Department of Electrical and Computer Engineering
    University of Houston, TX/USA
    
    This file is part of UHVision Libraries.
    
    UH Vision libraries are free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    UH Vison Libraries are distributed in the hope that it will be useful,
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
    Date:     2/20/17
    File:     controller_unit
    Comments: 
    **********************************************************************************
'''

import tensorflow as tf

import metalearning_tf.utils.tf_utils as tf_utils


class ControllerUnit(object):

    # Layer Names
    LAYER_HIDDEN = "cHidden"
    LAYER_OUTPUT = "cOut"

    # Node Names
    NODE_HIDDEN_W_XH = "w_xh"
    NODE_HIDDEN_W_RH = "w_rh"
    NODE_HIDDEN_W_HH = "w_hh"
    NODE_HIDDEN_BI = "bias_h"

    NODE_OUT_W_KEY = "w_key"
    NODE_OUT_W_ADD = "w_add"
    NODE_OUT_W_SIG = "w_sig"
    NODE_OUT_B_KEY = "bias_key"
    NODE_OUT_B_ADD = "bias_add"
    NODE_OUT_B_SIG = "bias_sig"

    # Config options
    __batch_size = 0
    __num_classes = 0
    __input_size = 0
    __controller_size = 0
    __memory_size = 0
    __num_read_heads = 0

    # Stateful tensors (Operational tensors). Changing over time

    # Controller state operations
    __h_0 = []
    __c_0 = []

    def __init__(self,
                 batch_size,
                 num_classes,
                 input_size,
                 controller_size,
                 memory_size,
                 num_read_heads):
        '''
        Constructor implementation for :class ControllerUnit

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

    def build(self):
        print('******************Building Controller Unit...')
        # Note that in this paper, they present previous output label as
        # input to the system; hence total number of inputs is equal to
        # "input_size + num_classes"
        num_inputs = self.__input_size+self.__num_classes
        print('Hidden Layer Nodes...')
        with tf.variable_scope(ControllerUnit.LAYER_HIDDEN, reuse=False):
            # input-->forget-->output-->approximated cell content
            # Weights associating input to the current hidden units
            init = tf_utils.glorot_uniform_init([num_inputs, 4 * self.__controller_size])
            tf.get_variable(ControllerUnit.NODE_HIDDEN_W_XH,
                            [num_inputs, 4 * self.__controller_size],
                            initializer=init)
            tf.get_variable(ControllerUnit.NODE_HIDDEN_BI,
                            [4 * self.__controller_size],
                            initializer=tf.zeros_initializer())
            # Weights associating previously read data from the memory to the current hidden units
            num_tot_reads = self.__num_read_heads*self.__memory_size[1]  # Number of total read weights
            init = tf_utils.glorot_uniform_init([num_tot_reads, 4 * self.__controller_size])
            tf.get_variable(ControllerUnit.NODE_HIDDEN_W_RH,
                            [num_tot_reads, 4 * self.__controller_size],
                            initializer=init)
            # Weights associating h_tm1 to the current hidden units
            init = tf_utils.glorot_uniform_init([self.__controller_size, 4 * self.__controller_size])
            tf.get_variable(ControllerUnit.NODE_HIDDEN_W_HH,
                            [self.__controller_size, 4 * self.__controller_size],
                            initializer=init)
        print('Creating Initial Controller State at time t=0...')
        self.__c_0 = tf.zeros([self.__batch_size, self.__controller_size], dtype=tf.float32)
        self.__h_0 = tf.zeros([self.__batch_size, self.__controller_size], dtype=tf.float32)
        print('Controller Output Nodes...')
        with tf.variable_scope(ControllerUnit.LAYER_OUTPUT):
            # Weight and Bias for 'key'
            init = tf_utils.glorot_uniform_init([self.__num_read_heads,
                                                 self.__memory_size[1]])
            tf.get_variable(ControllerUnit.NODE_OUT_W_KEY,
                            [self.__num_read_heads,
                             self.__controller_size,
                             self.__memory_size[1]],
                            initializer=init)
            tf.get_variable(ControllerUnit.NODE_OUT_B_KEY,
                            [self.__num_read_heads, self.__memory_size[1]],
                            initializer=tf.zeros_initializer())
            # Weight and Bias for 'add'
            init = tf_utils.glorot_uniform_init([self.__controller_size,
                                                 self.__memory_size[1]])
            tf.get_variable(ControllerUnit.NODE_OUT_W_ADD,
                            [self.__num_read_heads,
                             self.__controller_size,
                             self.__memory_size[1]],
                            initializer=init)
            tf.get_variable(ControllerUnit.NODE_OUT_B_ADD,
                            [self.__num_read_heads, self.__memory_size[1]],
                            initializer=tf.zeros_initializer())
            # Weight and Bias for 'sigma'
            init = tf_utils.glorot_uniform_init([self.__controller_size,
                                                 1])
            tf.get_variable(ControllerUnit.NODE_OUT_W_SIG,
                            [self.__num_read_heads, self.__controller_size, 1],
                            initializer=init)
            tf.get_variable(ControllerUnit.NODE_OUT_B_SIG,
                            [self.__num_read_heads, 1],
                            initializer=tf.zeros_initializer())
        print()
        print()

    def get_init_op_state(self):

        return self.__c_0, self.__h_0

    @staticmethod
    def get_node_out():
        '''
        Don't call before building is complete!!!

        :return:
        '''
        with tf.variable_scope(ControllerUnit.LAYER_OUTPUT) as out_scope:
            out_scope.reuse_variables()
            w_key = tf.get_variable(ControllerUnit.NODE_OUT_W_KEY)  # NRxCSxMIL
            b_key = tf.get_variable(ControllerUnit.NODE_OUT_B_KEY)  # NRxMIL
            w_add = tf.get_variable(ControllerUnit.NODE_OUT_W_ADD)  # NRxCSxMIL
            b_add = tf.get_variable(ControllerUnit.NODE_OUT_B_ADD)  # NRxMIL
            w_sig = tf.get_variable(ControllerUnit.NODE_OUT_W_SIG)  # NRxCSx1
            b_sig = tf.get_variable(ControllerUnit.NODE_OUT_B_SIG)  # NRx1
        return w_key, b_key, w_add, b_add, w_sig, b_sig

    @staticmethod
    def get_node_hidden():
        '''
        Don't call before building is complete!!!

        :return:
        '''
        with tf.variable_scope(ControllerUnit.LAYER_HIDDEN) as hidden_scope:
            hidden_scope.reuse_variables()
            bias_h = tf.get_variable(ControllerUnit.NODE_HIDDEN_BI)  # CSx1
            w_xh = tf.get_variable(ControllerUnit.NODE_HIDDEN_W_XH)  # (num_inputs)xCS
            w_rh = tf.get_variable(ControllerUnit.NODE_HIDDEN_W_RH)  # (NR.MIL)xCS
            w_hh = tf.get_variable(ControllerUnit.NODE_HIDDEN_W_HH)  # CSxCS
            return bias_h, w_xh, w_rh, w_hh

    @staticmethod
    def slice_equally(x, size, nb_slice):
        return [x[:, n * size:(n + 1) * size] for n in range(nb_slice)]

    def inference(self, x_t, c_tm1, h_tm1, r_tm1):
        '''
        Don't call before building is complete!!!

        :param x_t:  Current input data -- BSx(num_classes+input_size)
        :param c_tm1: Content of the previous cell at time t-1 -- BSxCS
        :param h_tm1: Previous output of the hidden layer at time t-1 -- BSxCS
        :param r_tm1: Previous output of the read head at time t-1
         (Memory meta_tf calculates it) -- BSx(NR.MIL)

        :return: None
        '''

        print('******************Executing Controller Unit Ops...')
        print('Hidden Layer Ops...')
        bias_h, w_xh, w_rh, w_hh = ControllerUnit.get_node_hidden()
        pre_activations = tf.add(tf.matmul(x_t, w_xh) +
                                 tf.matmul(r_tm1, w_rh) +
                                 tf.matmul(h_tm1, w_hh),
                                 bias_h)  # BSx(4.CS)
        [gate_i, gate_f, gate_o, approx] = ControllerUnit.slice_equally(pre_activations, self.__controller_size, 4)
        gate_i = tf.nn.sigmoid(gate_i)  # BSxCS
        gate_f = tf.nn.sigmoid(gate_f)  # BSxCS
        gate_o = tf.nn.sigmoid(gate_o)  # BSxCS
        approx = tf.nn.tanh(approx)  # BSxCS
        # End of for-Loop
        print('Controller State Ops...')
        c_t = tf.add(tf.multiply(gate_f, c_tm1),
                     tf.multiply(gate_i, approx))
        h_t = tf.multiply(gate_o, tf.nn.tanh(c_t))
        print('Controller Output Ops...')
        w_key, b_key, w_add, b_add, w_sig, b_sig = ControllerUnit.get_node_out()
        # Calculating key, add, and alpha (learnable sigmoid gate parameter)
        k_t = tf.matmul(h_t, tf.reshape(w_key, shape=(self.__controller_size,-1)))
        k_t = tf.tanh(tf.reshape(k_t, shape=(self.__batch_size,
                                             self.__num_read_heads,
                                             self.__memory_size[1])) + b_key)  # BSxNRxMIL
        a_t = tf.matmul(h_t, tf.reshape(w_add, shape=(self.__controller_size, -1)))
        a_t = tf.tanh(tf.reshape(a_t, shape=(self.__batch_size,
                                             self.__num_read_heads,
                                             self.__memory_size[1])) + b_add)  # BSxNRxMIL
        s_t = tf.matmul(h_t, tf.reshape(w_sig, shape=(self.__controller_size, -1)))
        s_t = tf.sigmoid(tf.reshape(s_t, shape=(self.__batch_size, self.__num_read_heads, 1)) + b_sig)  # BSxNRx1

        return c_t, h_t, k_t, a_t, s_t

