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
    Comments: 
    **********************************************************************************
'''

import numpy as np
import tensorflow as tf

import utils.tf_utils as tf_utils


class MemoryUnit(object):

    __m_t = []  # Memory content
    __r_t = []  # Data read from memory at time t
    __wr_t = []  # Reading weights based cosine similarity at time t
    __wu_t = []  # Usage weights based on LRU-access at time t

    # Config options
    __batch_size = 0
    __memory_size = 0
    __num_read_heads = 0

    def __init__(self,
                 batch_size,
                 memory_size,
                 num_read_heads):

        self.__batch_size = batch_size
        self.__memory_size = memory_size
        self.__num_read_heads = num_read_heads

    def inference(self):

        print('****************************************************************')
        print('**********************Building memory...\n**********************')
        print('****************************************************************')
        with tf.variable_scope("memory"):
            # Dimensionality of the memory: batch_size X (mem_length)x(mem_el_size)
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

    def update_mem(self, s_t, wr_tm1):
        '''
        Define operations to update memory and weights

        :param s_t: Learnable interpolation value at time t--BSxNRx1
        :param wr_tm1: Read-weights at time t-1--BSxNRxMS
        :return:
        '''

        print('******************************************************')
        print('****************Memory Update Ops...\n****************')
        print('******************************************************')
        with tf.variable_scope("update_mem"):
            # Indices of Least-used weights at time t-1
            wlu_tm1 = tf_utils.get_sorted_idx(self.__wu_t,
                                              self.__wu_t.get_shape().as_list(),
                                              self.__num_read_heads)  # BSxNR
            # Write-weights at time t
            # Broadcasting s_t via its 3rd dimension for its size is
            # BSxNRx1 while wr_tm1 is with the size of BSxNRxMS
            ww_t = tf.multiply(s_t, wr_tm1)  # BSxNRxMS
            ww_t = tf.reshape(ww_t, (self.__batch_size*self.__num_read_heads,
                                     self.__memory_size[0]))  # (BS.NR)xMS
            first_dim = tf.constant(np.arange(0,
                                              self.__batch_size*self.__num_read_heads,
                                              dtype=np.int32).tolist(),
                                    dtype=tf.int32)
            sec_dim = tf.reshape(wlu_tm1, (self.__batch_size*self.__num_read_heads))
            indices_list = [first_dim, sec_dim]
            addition = tf.subtract(1.0, tf.reshape(s_t, (self.__batch_size, self.__num_read_heads)))
            ww_t = tf_utils.inc_tensor_els(ww_t, indices_list, addition)
            # Before writing to memory, Least-used locations are erased.
