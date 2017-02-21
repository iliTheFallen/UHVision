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


class ControllerUnit(object):

    # Tensors for the state of LSTM cell
    __c_t = 0  # So called "State of the Cell" C_t
    __h_t = 0  # Output produced by the current Cell h_t
    # Tensors for the hidden layer
    __gate_out = {}
    __gate_params = {}
    __approx_cont = None
    __approx_cont_params = []
    # Tensors for the controller's output
    __c_out = {}
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

    def get_c_hl(self):
        '''

        :return: gate_out, gate_params, approx_cont, approx_cont_params
         -- Output of each gate, Weights and Biases assoc. with each gate,
         Approximated Content (C_t_hat), Weights and Biases assoc. with Approximated Content
        '''
        return self.__gate_out, \
               self.__gate_params, \
               self.__approx_cont, \
               self.__approx_cont_params

    def get_c_state(self):
        '''

        :return: h_t, c_t -- Controller state
        '''
        return self.__h_t, \
               self.__c_t

    def get_c_out(self):
        '''

        :return: key_t, add_t, sig_t -- Controller output
        '''
        return self.__c_out

    def inference(self, x_t, cell_tm1=None, r_tm1=None, h_tm1=None):
        '''

        :param x_t:  Current input data -- BSx(num_classes+input_size)
        :param cell_tm1: Content of the previous cell at time t-1 -- BSxCS
        :param r_tm1: Previous output of the read head at time t-1 -- BSx(NR.MIL)
        :param h_tm1: Previous output of the hidden layer at time t-1 -- BSxCS
        :return:
        '''

        print('****************************************************************')
        print('*********************Building controller...*********************')
        print('****************************************************************\n')
        # Note that in this paper, they present previous output label as
        # input to the system; hence total number of inputs is equal to
        # "input_size + num_classes"
        num_inputs = self.__input_size+self.__num_classes
        # *************************************Hidden Layer*************************************
        with tf.variable_scope("cHidden"):
            gate_params = {}  # Param tensors for each gate input-->forget-->output in order
            gate_out = {}  # Output tensors for each gate input-->forget-->output in order
            approx_cont = None  # Output tensor for approximated cell content
            approx_cont_params = []  # Param tensors for approximated cell content
            weights = []  # List of weights for each component
            for i in range(4):  # input-->forget-->output-->approximated cell content
                # Weights associating input to the current hidden units
                w_xh = tf.Variable(tf.truncated_normal([num_inputs, self.__controller_size],
                                                       stddev=1.0 / math.sqrt(float(num_inputs)),
                                                       dtype=tf.float32),
                                   name="w_xh_"+str(i))
                # Weights associating previously read data from the memory to the current hidden unit
                num_tot_reads = self.__num_read_heads*self.__memory_size[1]  # Number of total read weights
                w_rh = tf.Variable(tf.truncated_normal([num_tot_reads, self.__controller_size],
                                                       stddev=1.0 / math.sqrt(float(num_tot_reads)),
                                                       dtype=tf.float32),
                                   name="w_rh_"+str(i))
                # Weights associating output of the previous hidden unit to the current hidden unit
                w_hh = tf.Variable(tf.truncated_normal([self.__controller_size, self.__controller_size],
                                                       stddev=1.0 / math.sqrt(float(self.__controller_size)),
                                                       dtype=tf.float32),
                                   name="w_hh_"+str(i))
                # Single bias for each hidden unit
                biases = tf.Variable(tf.zeros([self.__controller_size], dtype=tf.float32), name="biases_"+str(i))
                out = tf.nn.sigmoid(tf.matmul(x_t, w_xh) +
                                    tf.matmul(r_tm1, w_rh) +
                                    tf.matmul(h_tm1, w_hh) +
                                    biases, name="sig_"+str(i))  # BSxCS
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
        self.__gate_out += gate_out
        self.__gate_params += gate_params
        self.__approx_cont += approx_cont
        self.__approx_cont_params += approx_cont_params
        # *************************************Controller State*************************************
        with tf.variable_scope("cState"):
            # State for the cell
            self.__c_t = gate_out[G_OUT + str(1)] * cell_tm1 + gate_out[G_OUT + str(0)] * approx_cont
            self.__h_t = gate_out[G_OUT+str(2)]*tf.nn.tanh(self.__c_t)
        # *************************************Controller Output Layer*************************************
        with tf.variable_scope('cOut'):
            w_key = tf.Variable(tf.truncated_normal([self.__num_read_heads,
                                                     self.__controller_size,
                                                     self.__memory_size[1]],
                                                    stddev=1.0 / math.sqrt(float(self.__controller_size)),
                                                    dtype=tf.float32),
                                name="w_key")
            b_key = tf.Variable(tf.zeros([self.__num_read_heads, self.__memory_size[1]]), name="bias_key")
            w_add = tf.Variable(tf.truncated_normal([self.__num_read_heads,
                                                     self.__controller_size,
                                                     self.__memory_size[1]],
                                                    stddev=1.0 / math.sqrt(float(self.__controller_size)),
                                                    dtype=tf.float32),
                                name="w_add")
            b_add = tf.Variable(tf.zeros([self.__num_read_heads, self.__memory_size[1]]), name="bias_add")
            w_sig = tf.Variable(tf.truncated_normal([self.__num_read_heads,
                                                     self.__controller_size,
                                                     1],
                                                    stddev=1.0 / math.sqrt(float(self.__controller_size)),
                                                    dtype=tf.float32),
                                name="w_sig")
            b_sig = tf.Variable(tf.zeros([self.__num_read_heads, 1]), name="bias_sig")
            # Calculating key, add, and alpha (learnable sigmoid gate parameter)
            k_t = tf.nn.tanh(tf.add(tf.tensordot(self.__h_t, w_key, axes=[[1], [1]]), b_key))  # BSxNRxMIL
            a_t = tf.nn.tanh(tf.add(tf.tensordot(self.__h_t, w_add, axes=[[1], [1]]), b_add))  # BSxNRxMIL
            s_t = tf.nn.sigmoid(tf.add(tf.tensordot(self.__h_t, w_sig, axes=[[1], [1]]), b_sig))  # BSxNRx1
            self.__c_out['k_t'] = k_t
            self.__c_out['a_t'] = a_t
            self.__c_out['s_t'] = s_t

