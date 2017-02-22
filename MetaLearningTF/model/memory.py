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
    File:     memory
    Comments: External Memory Unit. In a physical sense, it might be conceived as a
      hardware component holding reader/writer heads, memory, and the bus-line
      between them.
    **********************************************************************************
'''

import numpy as np
import tensorflow as tf

import utils.tf_utils as tf_utils
import utils.similarity_measures as sim_utils


class MemoryUnit(object):

    # Tensors for storing state of the Memory
    __m_t = []  # Memory content
    __r_t = []  # Data read from memory at time t
    __wr_t = []  # Reading weights based cosine similarity at time t
    __wu_t = []  # Usage weights based on LRU-access at time t
    # Constants used throughout the application
    __m_eraser = None  # Used to erase the memory
    __ww_t_1D = None  # Used to index the 1st dimension of ww_t
    __m_tm1_1D = None  # Used to index the 1st dimension of m_tm1
    __gamma = 0  # Used to decay usage weights at time t-1

    # Config options
    __batch_size = 0
    __memory_size = 0
    __num_read_heads = 0

    def __init__(self,
                 batch_size,
                 memory_size,
                 num_read_heads,
                 gamma):

        self.__batch_size = batch_size
        self.__memory_size = memory_size
        self.__num_read_heads = num_read_heads
        # Create constant tensors
        self.__m_eraser = tf.zeros((self.__batch_size,
                                    self.__batch_size,
                                    self.__memory_size[1]),
                                   dtype=tf.float32)
        self.__ww_t_1D = tf.constant(np.arange(0,
                                               self.__batch_size * self.__num_read_heads,
                                               dtype=np.int32).tolist(),
                                     dtype=tf.int32)
        self.__m_tm1_1D = tf.constant(np.arange(0,
                                                self.__batch_size,
                                                dtype=np.int32).tolist(),
                                      dtype=tf.int32)
        self.__gamma = tf.constant([gamma], dtype=tf.float32, shape=[1], name="gamma")

    def get_m_t(self):
        return self.__m_t

    def get_r_t(self):
        return self.__r_t

    def get_wr_t(self):
        return self.__wr_t

    def get_wu_t(self):
        return self.__wu_t

    def inference(self):

        print('****************************************************************')
        print('**********************Building memory...\n**********************')
        print('****************************************************************')
        with tf.variable_scope("memory"):
            # Dimensionality of the memory: BS X (MS)x(MIL)
            # I.e. it has batch_size # of memories, each of which has a size of
            # (mem_length)x(mem_el_size)
            self.__m_t = tf.Variable(tf.multiply(1e-6, tf.ones((self.__batch_size,)+self.__memory_size,
                                                               dtype=tf.float32)),
                                     name="m_t")
            self.__r_t = tf.Variable(tf.zeros([self.__batch_size,
                                               self.__num_read_heads*self.__memory_size[1]]),
                                     name="r_t")
            # 1st memory-loc for each read-head is on / rest is off
            self.__wr_t = tf_utils.create_one_hot_var(
                (self.__batch_size, self.__num_read_heads),
                self.__memory_size[0],
                name="wr_t")
            self.__wu_t = tf_utils.create_one_hot_var(
                self.__batch_size,
                self.__memory_size[0],
                name="wu_t")

    def update_mem(self, m_tm1, wr_tm1, wu_tm1, s_t, a_t, k_t):
        '''
        Define operations to update memory and weights

        :param m_tm1: State of the memory at time t-1
        :param wr_tm1: Read-weights at time t-1--BSxNRxMS
        :param wu_tm1: Usage-weights at time t-1--
        :param s_t: Learnable interpolation value at time t--BSxNRx1
        :param a_t: Information per sample stored in the memory--BSxNRxMIL
        :param k_t: Key produced by the controller to generate read-weights--BSxNRxMIL
        :return:
        '''

        print('******************************************************')
        print('****************Memory Update Ops...\n****************')
        print('******************************************************')
        with tf.variable_scope("update_mem"):
            # Indices of Least-used weights at time t-1
            wlu_tm1 = tf_utils.get_sorted_idx(wu_tm1,
                                              self.__wu_t.get_shape().as_list(),
                                              self.__num_read_heads)  # BSxNR
            # Write-weights at time t
            # Broadcasting s_t via its 3rd dimension for its size is
            # BSxNRx1 while wr_tm1 is with the size of BSxNRxMS
            ww_t = tf.multiply(s_t, wr_tm1)  # BSxNRxMS
            ww_t = tf.reshape(ww_t, (self.__batch_size*self.__num_read_heads,
                                     self.__memory_size[0]))  # (BS.NR)xMS
            sec_dim = tf.reshape(wlu_tm1, (self.__batch_size*self.__num_read_heads))
            indices_list = [self.__ww_t_1D, sec_dim]
            addition = tf.subtract(1.0, tf.reshape(s_t,
                                                   (self.__batch_size,
                                                    self.__num_read_heads)))  # BSxNR
            ww_t = tf_utils.inc_tensor_els(ww_t, indices_list, addition)
            # Write-weights at time t after interpolation of
            # read-weights and least-used weights at time t-1
            ww_t = tf.reshape(ww_t, (self.__batch_size,
                                     self.__num_read_heads,
                                     self.__memory_size[0]))  # BSxNRxMS
            # Before writing to memory, Least-used locations are erased.
            sec_dim = tf.slice(wlu_tm1, [0, 0], [self.__batch_size, 1])
            indices_list = [self.__m_tm1_1D, sec_dim]
            tf_utils.update_tensor_els(m_tm1, indices_list, self.__m_eraser)
            # Write the information produced by the controller
            # after weighting it with write-weights at time t.
            # Each memory location has a weight that corresponds to each read head.
            # Information read by each head stored in each row (2nd dimension)
            # of a_t for a given sample (1st dimension).
            w_a_t = tf.matmul(tf.transpose(ww_t, perm=[0, 2, 1]), a_t)  # batch_mul(BSxMSxNR, BSxNRxMIL)
            self.__m_t = tf.add(self.__m_t, w_a_t)  # BSxMSxMIL
            # Read weights at time t
            K_t = sim_utils.cosine_similarity(k_t, self.__m_t)  # BSxNRxMS
            self.__wr_t = tf.nn.softmax(tf.reshape(K_t, (self.__batch_size*self.__num_read_heads,
                                                         self.__memory_size[0])))
            self.__wr_t = tf.reshape(self.__wr_t, (self.__batch_size,
                                                   self.__num_read_heads,
                                                   self.__memory_size[0]))
            # Memory Usage weights at time t
            self.__wu_t = tf.add(tf.multiply(self.__gamma, wu_tm1),
                                 tf.reduce_sum(self.__wr_t, axis=1))
            self.__wu_t = tf.add(self.__wu_t,
                                 tf.reduce_sum(ww_t, axis=1))  # BSxMS
            # Use read-heads to read data from memory
            self.__r_t = tf.reshape(tf.matmul(self.__wr_t, self.__m_t), [self.__batch_size, -1])
