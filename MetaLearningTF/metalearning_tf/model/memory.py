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
    Date:     2/20/17
    File:     memory
    Comments: External Memory Unit. In a physical sense, it might be conceived as a
      hardware component holding reader/writer heads, memory, and the bus-line
      between them.
    **********************************************************************************
'''

import tensorflow as tf
import numpy as np
from metalearning_tf.utils import tf_utils

import metalearning_tf.utils.similarity_measures as sim_utils


class MemoryUnit(object):

    LAYER_UPDATE_MEM = "updateMem"

    # Config options
    __batch_size = 0
    __memory_size = 0
    __num_read_heads = 0

    # Constant tensors
    __ww_t_1d = None  # Used to index the 1st dimension of ww_t
    __m_tm1_1d = None   # Used to index the 1st dimension of m_tm1
    __m_eraser = None  # Used to erase the memory
    __gamma = 0  # Used to decay usage weights at time t-1

    # Stateful tensors (Operational tensors). Changing over time

    # Memory operations
    __m_0 = None
    __r_0 = None
    __wr_0 = None
    __wu_0 = None

    # Variables used as references
    __temp_m_t = None
    __ww_t = None

    def __init__(self,
                 batch_size,
                 memory_size,
                 num_read_heads,
                 gamma):

        self.__batch_size = batch_size
        self.__memory_size = memory_size
        self.__num_read_heads = num_read_heads
        self.__gamma = gamma

    def build(self):
        print('******************Building Memory...')
        # Create constant tensors
        print('Building Memory Operation Constants...')
        self.__m_eraser = tf.zeros((self.__batch_size,
                                    self.__memory_size[1]),
                                   name="memoryEraser")
        self.__ww_t_1d = tf.constant(np.arange(0,
                                               self.__batch_size * self.__num_read_heads,
                                               dtype=np.int32).tolist(),
                                     dtype=tf.int32)
        self.__m_tm1_1d = tf.range(0, self.__batch_size, dtype=tf.int32, name="memoryUp1stDim")
        self.__gamma = tf.constant([self.__gamma], shape=[1], dtype=tf.float32, name="gamma")
        # Create operational tensors.
        # Dimensionality of the memory: BSx(MS)x(MIL)
        # I.e. it has batch_size # of memories, each of which has a size of
        # (mem_length)x(mem_el_size)
        print('Building Memory Update Operations...')
        self.__m_0 = tf.multiply(tf.to_float(1e-6),
                                 tf.ones((self.__batch_size,)+self.__memory_size, dtype=tf.float32))
        # 1st memory-loc for each read-head is on / rest is off. Both are non-trainable tensors
        self.__wr_0 = tf_utils.create_one_hot_var((self.__batch_size, self.__num_read_heads),
                                                  self.__memory_size[0])
        self.__wu_0 = tf_utils.create_one_hot_var(self.__batch_size, self.__memory_size[0])
        self.__r_0 = tf.zeros([self.__batch_size, self.__num_read_heads*self.__memory_size[1]], dtype=tf.float32)
        print()
        print()

    def get_init_op_memory(self):

        return self.__m_0, self.__r_0, self.__wr_0, self.__wu_0

    def update_mem(self, m_tm1, wr_tm1, wu_tm1, k_t, a_t, s_t):
        '''
        Define operations to update memory and weights

        :param m_tm1: State of the memory at time t-1
        :param wr_tm1: Read-weights at time t-1--BSxNRxMS
        :param wu_tm1: Usage-weights at time t-1-- BSxMS
        :param k_t: Key produced by the controller to generate read-weights--BSxNRxMIL
        :param a_t: Information per sample that will be stored in the memory--BSxNRxMIL
        :param s_t: Learnable interpolation value at time t--BSxNRx1
        :return:
        '''

        print('******************Executing Memory Update Ops...')
        # Where m_t is updated
        print('Updating Memory...')
        with tf.variable_scope(MemoryUnit.LAYER_UPDATE_MEM):
            # Indices of Least-used weights at time t-1
            wlu_tm1 = tf_utils.get_sorted_idx(wu_tm1,
                                              wu_tm1.get_shape().as_list(),
                                              self.__num_read_heads,
                                              is_ascending=True)  # BSxNR
            # Write-weights at time t
            # Broadcasting s_t via its 3rd dimension for its size is
            # BSxNRx1 while wr_tm1 is with the size of BSxNRxMS
            ww_t = tf.multiply(s_t, wr_tm1)  # BSxNRxMS
            ww_t = tf.reshape(ww_t, [self.__batch_size*self.__num_read_heads,
                                     self.__memory_size[0]])  # (BS.NR)xMS
            sec_dim = tf.reshape(wlu_tm1, [-1])  # BS.NR
            addition = tf.subtract(tf.constant([1.0]), tf.reshape(s_t, [-1]))  # BS.NR
            ww_t = tf_utils.inc_tensor_els(ww_t,
                                           addition,
                                           [self.__ww_t_1d, sec_dim])
            # Write-weights at time t after interpolation of
            # read-weights and least-used weights at time t-1
            ww_t = tf.reshape(ww_t, [self.__batch_size,
                                     self.__num_read_heads,
                                     self.__memory_size[0]])  # BSxNRxMS
            # Before writing to memory, Least-used locations are erased.
            sec_dim = wlu_tm1[:, 0]
            m_t = tf_utils.update_var_els(m_tm1,
                                          self.__m_eraser,
                                          [self.__m_tm1_1d, sec_dim])
            # Write the information produced by the controller
            # after weighting it with write-weights at time t.
            # Each memory location has a weight that corresponds to each read head.
            # Information read by each head stored in each row (2nd dimension)
            # of a_t for a given sample (1st dimension).
            w_a_t = tf.matmul(tf.transpose(ww_t, perm=[0, 2, 1]), a_t)  # batch_mul(BSxMSxNR, BSxNRxMIL)
            m_t = tf.add(m_t, w_a_t)  # BSxMSxMIL
        print('Updating Weights (Read & Usage)...')
        # Update for the next step
        K_t = sim_utils.cosine_similarity(k_t, m_t)  # BSxNRxMS
        wr_t = tf.nn.softmax(tf.reshape(K_t, [self.__batch_size*self.__num_read_heads,
                                              self.__memory_size[0]]))
        wr_t = tf.reshape(wr_t, [self.__batch_size, self.__num_read_heads, self.__memory_size[0]])  # BSxNRxMS
        # Memory Usage weights at time t
        wu_t = tf.add(tf.multiply(self.__gamma, wu_tm1),
                      tf.reduce_sum(wr_t, axis=1))
        wu_t = tf.add(wu_t, tf.reduce_sum(ww_t, axis=1))  # BSxMS
        # Use read-heads to read data from memory
        print('Reading data from memory...')
        r_t = tf.reshape(tf.matmul(wr_t, m_t), [self.__batch_size, -1])  # BSx(NR.MIL)

        return m_t, r_t, wr_t, wu_t
