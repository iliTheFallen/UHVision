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

import math
import tensorflow as tf


class ControllerUnit(object):

    G_OUT = 'gOut_'
    PARAM_MAP = 'paramMap_'

    # Layer Names
    LAYER_HIDDEN = "cHidden"
    LAYER_HIDDEN_OPS = "cHiddenOps"

    LAYER_OUTPUT = "cOut"
    LAYER_OUTPUT_OPS = 'cOutOps'

    LAYER_STATE = "cState"

    # Node Names
    NODE_HIDDEN_W_XH = "w_xh_"
    NODE_HIDDEN_W_RH = "w_rh_"
    NODE_HIDDEN_W_HH = "w_hh_"
    NODE_HIDDEN_BI = "bias_h"

    NODE_OUT_W_KEY = "w_key"
    NODE_OUT_W_ADD = "w_add"
    NODE_OUT_W_SIG = "w_sig"
    NODE_OUT_B_KEY = "bias_key"
    NODE_OUT_B_ADD = "bias_add"
    NODE_OUT_B_SIG = "bias_sig"

    # Operation Names
    OP_GATE = "op_gate_"
    OP_APPROX = "op_approx"

    OP_H_T = "h_t"
    OP_C_T = "c_t"

    OP_K_T = "k_t"
    OP_A_T = "a_t"
    OP_S_T = "s_t"

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
        # Note that in this paper, they present previous output label as
        # input to the system; hence total number of inputs is equal to
        # "input_size + num_classes"
        num_inputs = self.__input_size+self.__num_classes
        print('****************************************************************')
        print('*********************Hidden Layer Nodes...**********************')
        print('****************************************************************')
        with tf.variable_scope(ControllerUnit.LAYER_HIDDEN):
            for i in range(4):  # input-->forget-->output-->approximated cell content
                # Weights associating input to the current hidden units
                tf.Variable(tf.truncated_normal([num_inputs, self.__controller_size],
                                                stddev=1.0/math.sqrt(float(num_inputs))),
                            name=ControllerUnit.NODE_HIDDEN_W_XH+str(i))
                # Weights associating previously read data from the memory to the current hidden unit
                num_tot_reads = self.__num_read_heads*self.__memory_size[1]  # Number of total read weights
                tf.Variable(tf.truncated_normal([num_tot_reads, self.__controller_size],
                                                stddev=1.0/math.sqrt(float(num_tot_reads))),
                            name=ControllerUnit.NODE_HIDDEN_W_RH+str(i))
                # Weights associating output of the previous hidden unit to the current hidden unit
                tf.Variable(tf.truncated_normal([self.__controller_size, self.__controller_size],
                                                stddev=1.0/math.sqrt(float(self.__controller_size))),
                            name=ControllerUnit.NODE_HIDDEN_W_HH+str(i))
                # Single bias for each hidden unit
            tf.Variable(tf.zeros([self.__controller_size]), name=ControllerUnit.NODE_HIDDEN_BI)
        print('****************************************************************')
        print('******************Controller Output Nodes...********************')
        print('****************************************************************')
        with tf.variable_scope(ControllerUnit.LAYER_OUTPUT):
            # Weight and Bias for 'key'
            tf.Variable(tf.truncated_normal([self.__num_read_heads,
                                             self.__controller_size,
                                             self.__memory_size[1]],
                                            stddev=1.0 / math.sqrt(
                                                float(self.__num_read_heads*self.__controller_size))),
                        name=ControllerUnit.NODE_OUT_W_KEY)
            tf.Variable(tf.zeros([self.__num_read_heads, self.__memory_size[1]]),
                        name=ControllerUnit.NODE_OUT_B_KEY)
            # Weight and Bias for 'add'
            tf.Variable(tf.truncated_normal([self.__num_read_heads,
                                             self.__controller_size,
                                             self.__memory_size[1]],
                                            stddev=1.0 / math.sqrt(
                                                float(self.__num_read_heads*self.__controller_size))),
                        name=ControllerUnit.NODE_OUT_W_ADD)
            tf.Variable(tf.zeros([self.__controller_size]),
                        name=ControllerUnit.NODE_OUT_B_ADD)
            # Weight and Bias for 'sigma'
            tf.Variable(tf.truncated_normal([self.__num_read_heads,
                                             self.__controller_size,
                                             1],
                                            stddev=1.0 / math.sqrt(float(self.__num_read_heads))),
                        name=ControllerUnit.NODE_OUT_W_SIG)
            tf.Variable(tf.zeros([self.__num_read_heads, 1]),
                        name=ControllerUnit.NODE_OUT_B_SIG)
        print()
        print()

    @staticmethod
    def get_op_hidden():
        with tf.variable_scope(ControllerUnit.LAYER_HIDDEN_OPS):
            return list(tf.get_variable(ControllerUnit.OP_GATE + str(i)) for i in range(3))

    @staticmethod
    def get_op_approx():
        with tf.variable_scope(ControllerUnit.LAYER_HIDDEN_OPS):
            return tf.get_variable(ControllerUnit.OP_APPROX)

    @staticmethod
    def get_op_state():
        with tf.variable_scope(ControllerUnit.LAYER_STATE):
            return tf.get_variable(ControllerUnit.OP_H_T), tf.get_variable(ControllerUnit.OP_C_T)

    @staticmethod
    def get_op_out():
        with tf.variable_scope(ControllerUnit.LAYER_OUTPUT_OPS):
            return tf.get_variable(ControllerUnit.OP_K_T), \
                   tf.get_variable(ControllerUnit.OP_A_T), \
                   tf.get_variable(ControllerUnit.OP_S_T)

    @staticmethod
    def get_node_out():
        '''
        Don't call before building is complete!!!

        :return:
        '''
        with tf.variable_scope(ControllerUnit.LAYER_OUTPUT) as out_scope:
            out_scope.reuse_variables()
            w_key = tf.get_variable(ControllerUnit.NODE_OUT_W_KEY)
            b_key = tf.get_variable(ControllerUnit.NODE_OUT_B_KEY)
            w_add = tf.get_variable(ControllerUnit.NODE_OUT_W_ADD)
            b_add = tf.get_variable(ControllerUnit.NODE_OUT_B_ADD)
            w_sig = tf.get_variable(ControllerUnit.NODE_OUT_W_SIG)
            b_sig = tf.get_variable(ControllerUnit.NODE_OUT_B_SIG)
        return w_key, b_key, w_add, b_add, w_sig, b_sig

    @staticmethod
    def get_node_hidden(gate_idx):
        '''
        Don't call before building is complete!!!

        :return:
        '''
        with tf.variable_scope(ControllerUnit.LAYER_HIDDEN) as hidden_scope:
            hidden_scope.reuse_variables()
            bias_h = tf.get_variable(ControllerUnit.NODE_HIDDEN_BI)
            w_xh = tf.get_variable(ControllerUnit.NODE_HIDDEN_W_XH + str(gate_idx))
            w_rh = tf.get_variable(ControllerUnit.NODE_HIDDEN_W_RH + str(gate_idx))
            w_hh = tf.get_variable(ControllerUnit.NODE_HIDDEN_W_HH + str(gate_idx))
            return bias_h, w_xh, w_rh, w_hh

    @staticmethod
    def inference(x_t, c_tm1, r_tm1, h_tm1):
        '''
        Don't call before building is complete!!!

        :param x_t:  Current input data -- BSx(num_classes+input_size)
        :param c_tm1: Content of the previous cell at time t-1 -- BSxCS
        :param r_tm1: Previous output of the read head at time t-1 -- BSx(NR.MIL)
        :param h_tm1: Previous output of the hidden layer at time t-1 -- BSxCS
        :return:
        '''
        print('******************************************************')
        print('*****************Hidden Layer Ops...******************')
        print('******************************************************\n')
        for i in range(4):
            with tf.variable_scope(ControllerUnit.LAYER_HIDDEN_OPS):
                bias_h, w_xh, w_rh, w_hh = ControllerUnit.get_node_hidden(i)
                if i < 3:
                    tf.nn.sigmoid(tf.matmul(x_t, w_xh) +
                                  tf.matmul(r_tm1, w_rh) +
                                  tf.matmul(h_tm1, w_hh) +
                                  bias_h,
                                  name=ControllerUnit.OP_GATE + str(i))  # BSxCS
                else:
                    tf.nn.tanh(tf.matmul(x_t, w_xh) +
                               tf.matmul(r_tm1, w_rh) +
                               tf.matmul(h_tm1, w_hh) +
                               bias_h,
                               name=ControllerUnit.OP_APPROX)  # BSxCS
        # End of for-Loop
        print('******************************************************')
        print('***************Controller State Ops...****************')
        print('******************************************************\n')
        [gate_i, gate_f, gate_o] = ControllerUnit.get_op_hidden()
        approx = ControllerUnit.get_op_approx()
        with tf.variable_scope(ControllerUnit.LAYER_STATE):
            tf.add(tf.multiply(gate_f, c_tm1),
                   tf.multiply(gate_i, approx),
                   name=ControllerUnit.OP_C_T)
            c_t = tf.get_variable(ControllerUnit.OP_C_T)
            tf.multiply(gate_o, tf.nn.tanh(c_t),
                        name=ControllerUnit.OP_H_T)
        print('******************************************************')
        print('**************Controller Output Ops...****************')
        print('******************************************************')
        [w_key, b_key, w_add, b_add, w_sig, b_sig] = ControllerUnit.get_node_out()
        [h_t, _] = ControllerUnit.get_op_state()
        with tf.variable_scope(ControllerUnit.LAYER_OUTPUT_OPS):
            # Calculating key, add, and alpha (learnable sigmoid gate parameter)
            tf.nn.tanh(tf.add(tf.tensordot(h_t, w_key, axes=[[1], [1]]), b_key),
                       name=ControllerUnit.OP_K_T)  # BSxNRxMIL
            tf.nn.tanh(tf.add(tf.tensordot(h_t, w_add, axes=[[1], [1]]), b_add),
                       name=ControllerUnit.OP_A_T)  # BSxNRxMIL
            tf.nn.sigmoid(tf.add(tf.tensordot(h_t, w_sig, axes=[[1], [1]]), b_sig),
                          name=ControllerUnit.OP_S_T)  # BSxNRx1
        print()
        print()

