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

    LAYER_M_CONSTS = "mConsts"
    LAYER_MEMORY_OPS = "memory"
    LAYER_UPDATE_MEM = "updateMem"

    # Node Names
    NODE_M_ERASER = "memEraser"  # Used to erase the memory
    NODE_M_WW_T_1D = "memWWT1D"  # Used to index the 1st dimension of ww_t
    NODE_M_TM1_1D = "memTM11D"  # Used to index the 1st dimension of m_tm1
    NODE_GAMMA = "gamma"  # Used to decay usage weights at time t-1

    # Operations
    OP_M_T = "m_t"  # Memory content
    OP_R_T = "r_t"  # Data read from memory at time t
    OP_WR_T = "wr_t"  # Reading weights based cosine similarity at time t
    OP_WU_T = "wu_t"  # Usage weights based on LRU-access at time t

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
        self.__gamma = gamma

    def build(self):
        print('****************************************************************')
        print('***********************Building memory...***********************')
        print('****************************************************************\n')
        # Create constant tensors
        with tf.variable_scope(MemoryUnit.LAYER_M_CONSTS):
            tf.zeros((self.__batch_size,
                      self.__batch_size,
                      self.__memory_size[1]),
                     dtype=tf.float32,
                     name=MemoryUnit.NODE_M_ERASER)
            tf.constant(np.arange(0,
                                  self.__batch_size * self.__num_read_heads,
                                  dtype=np.int32).tolist(),
                        dtype=tf.int32,
                        name=MemoryUnit.NODE_M_WW_T_1D)
            tf.constant(np.arange(0,
                                  self.__batch_size,
                                  dtype=np.int32).tolist(),
                        dtype=tf.int32,
                        name=MemoryUnit.NODE_M_TM1_1D)
            tf.constant([self.__gamma], dtype=tf.float32, shape=[1], name=MemoryUnit.NODE_GAMMA)
        with tf.variable_scope(MemoryUnit.LAYER_MEMORY_OPS):
            # Dimensionality of the memory: BS X (MS)x(MIL)
            # I.e. it has batch_size # of memories, each of which has a size of
            # (mem_length)x(mem_el_size)
            tf.Variable(tf.multiply(1e-6,
                                    tf.ones((self.__batch_size,)+self.__memory_size,
                                            dtype=tf.float32)),
                        name=MemoryUnit.OP_M_T)
            tf.Variable(tf.zeros([self.__batch_size,
                                  self.__num_read_heads*self.__memory_size[1]]),
                        name=MemoryUnit.OP_R_T)
            # 1st memory-loc for each read-head is on / rest is off
            tf_utils.create_one_hot_var((self.__batch_size, self.__num_read_heads),
                                        self.__memory_size[0],
                                        name=MemoryUnit.OP_WR_T)
            tf_utils.create_one_hot_var(self.__batch_size,
                                        self.__memory_size[0],
                                        name=MemoryUnit.OP_WU_T)

    @staticmethod
    def get_node_m_consts():
        with tf.variable_scope(MemoryUnit.LAYER_M_CONSTS) as const_scope:
            const_scope.reuse_variables()
            return tf.get_variable(MemoryUnit.NODE_M_ERASER), \
                   tf.get_variable(MemoryUnit.NODE_M_WW_T_1D), \
                   tf.get_variable(MemoryUnit.NODE_M_TM1_1D), \
                   tf.get_variable(MemoryUnit.NODE_GAMMA)

    @staticmethod
    def get_op_memory():
        with tf.variable_scope(MemoryUnit.LAYER_MEMORY_OPS):
            m_t = tf.get_variable(MemoryUnit.OP_M_T)
            r_t = tf.get_variable(MemoryUnit.OP_R_T)
            wr_t = tf.get_variable(MemoryUnit.OP_WR_T)
            wu_t = tf.get_variable(MemoryUnit.OP_WU_T)
        return m_t, r_t, wr_t, wu_t

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
        print('*****************Memory Update Ops...*****************')
        print('******************************************************')
        [m_eraser, ww_t_1d, m_tm1_1d, gamma] = MemoryUnit.get_node_m_consts()
        [m_t, r_t, wr_t, wu_t] = MemoryUnit.get_op_memory()
        # Where m_t is updated
        with tf.variable_scope(MemoryUnit.LAYER_UPDATE_MEM):
            # Indices of Least-used weights at time t-1
            wlu_tm1 = tf_utils.get_sorted_idx(wu_tm1,
                                              wu_t.get_shape().as_list(),
                                              self.__num_read_heads)  # BSxNR
            # Write-weights at time t
            # Broadcasting s_t via its 3rd dimension for its size is
            # BSxNRx1 while wr_tm1 is with the size of BSxNRxMS
            ww_t = tf.multiply(s_t, wr_tm1)  # BSxNRxMS
            ww_t = tf.reshape(ww_t, (self.__batch_size*self.__num_read_heads,
                                     self.__memory_size[0]))  # (BS.NR)xMS
            sec_dim = tf.reshape(wlu_tm1, (self.__batch_size*self.__num_read_heads))
            indices_list = [ww_t_1d, sec_dim]
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
            indices_list = [m_tm1_1d, sec_dim]
            tf_utils.update_tensor_els(m_tm1, indices_list, m_eraser)
            # Write the information produced by the controller
            # after weighting it with write-weights at time t.
            # Each memory location has a weight that corresponds to each read head.
            # Information read by each head stored in each row (2nd dimension)
            # of a_t for a given sample (1st dimension).
            w_a_t = tf.matmul(tf.transpose(ww_t, perm=[0, 2, 1]), a_t)  # batch_mul(BSxMSxNR, BSxNRxMIL)
            tf.assign(m_t, tf.add(m_t, w_a_t))  # BSxMSxMIL

        # Update for the next step
        K_t = sim_utils.cosine_similarity(k_t, m_t)  # BSxNRxMS
        tf.assign(wr_t,
                  tf.nn.softmax(tf.reshape(K_t, (self.__batch_size*self.__num_read_heads,
                                                 self.__memory_size[0]))))
        tf.assign(wr_t,
                  tf.reshape(wr_t, (self.__batch_size,
                                    self.__num_read_heads,
                                    self.__memory_size[0])))
        # Memory Usage weights at time t
        tf.assign(wu_t, tf.add(tf.multiply(gamma, wu_tm1), tf.reduce_sum(wr_t, axis=1)))
        tf.assign(wu_t, tf.add(wu_t, tf.reduce_sum(ww_t, axis=1)))  # BSxMS
        # Use read-heads to read data from memory
        tf.assign(r_t, tf.reshape(tf.matmul(wr_t, m_t), [self.__batch_size, -1]))  # BSx(NR.MIL)
