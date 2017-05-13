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
    File:     nn_model
    Comments: This file contains the architecture for NTM which is based
     on the memory-access model defined in the paper called
     "Meta-Learning with Memory-Augmented Neural Networks".
    **********************************************************************************
'''

import tensorflow as tf

from metalearning_tf.model.base_model import BaseModel
from metalearning_tf.model.memory import MemoryUnit
from metalearning_tf.model.controller_unit import ControllerUnit
from metalearning_tf.utils import loss_funcs
import metalearning_tf.utils.tf_utils as tf_utils


class NNModel(BaseModel):
    # Layers
    LAYER_NN_OUT = "nnOut"

    # Nodes
    NODE_W_OUT = "w_o"
    NODE_B_OUT = "b_o"

    # Configuration options
    __is_for_regress = None  # Whether network is built for regression or classification
    __batch_size = None  # Number of samples fed into network at a particular time t
    __seq_len = None  # How many times RNN is unrolled
    __num_classes = None  # Number of classes/labels
    __controller_size = None  # How many hidden units the RNN cell has
    __memory_size = None  # 2D Tuple specifying (numberOfElements)X(sizeOfEachElement)
    __num_read_heads = None  # Number of read heads attached to controller
    __learning_rate = None  # Learning rate passed directly to the optimizer
    __percentile = None  # The fraction of all samples to which we apply LAD when calculating loss

    # Two main components that this nn_model tries to fuse one with another
    __controller = None
    __memory = None

    # Used for chain call of generate_model(), loss(), train()
    __labels = None
    __preds = None
    __loss = None
    __train_op = None

    def __init__(self,
                 inputs=None,
                 targets=None,
                 is_for_regress=False,
                 batch_size=16,
                 seq_len=50,
                 input_size=20*20,
                 num_classes=5,
                 controller_size=200,
                 memory_size=(128, 40),
                 num_read_heads=4,
                 learning_rate=0.001,
                 percentile=0.5,
                 gamma=0.95):

        self.__is_for_regress = is_for_regress
        self.__batch_size = batch_size
        self.__seq_len = seq_len
        self.__num_classes = num_classes
        self.__controller_size = controller_size
        self.__memory_size = memory_size
        self.__num_read_heads = num_read_heads
        self.__learning_rate = learning_rate
        self.__percentile = percentile
        # Create main components
        # TODO: What if offset input is disabled by overriding _offset_input method.
        self.__controller = ControllerUnit(batch_size,
                                           num_classes,
                                           input_size+num_classes,  # Offset input
                                           controller_size,
                                           memory_size,
                                           num_read_heads)
        self.__memory = MemoryUnit(batch_size,
                                   memory_size,
                                   num_read_heads,
                                   gamma)
        input_ph_size = (self.__batch_size,
                         self.__seq_len,
                         input_size)
        input_type = tf.float32
        target_size = (self.__batch_size,
                       self.__seq_len,
                       self.__num_classes)
        if is_for_regress:
            target_type = tf.float32
        else:
            target_type = tf.int32
        super(NNModel, self).__init__(input_ph_size,
                                      target_size,
                                      input_type,
                                      target_type,
                                      inputs=inputs,
                                      targets=targets)

    # TODO: Add sanity check for whether the provided arguments are valid
    # TODO: in the specified mode (classification OR regression)

    def _create_output_nodes(self):
        '''

        :return: None
        '''

        print('Creating output weights and bias...')
        num_weights = self.__controller_size+self.__num_read_heads*self.__memory_size[1]
        with tf.variable_scope(NNModel.LAYER_NN_OUT):
            init = tf_utils.glorot_uniform_init([num_weights, self.__num_classes])
            tf.get_variable(NNModel.NODE_W_OUT,
                            [num_weights,
                             self.__num_classes],
                            initializer=init)
            tf.get_variable(NNModel.NODE_B_OUT,
                            [self.__num_classes],
                            initializer=tf.zeros_initializer())

    @staticmethod
    def _get_node_out():
        '''
        Don't call before building is complete!!!

        :return:
        '''
        with tf.variable_scope(NNModel.LAYER_NN_OUT) as out_scope:
            out_scope.reuse_variables()
            w_o = tf.get_variable(NNModel.NODE_W_OUT)
            b_o = tf.get_variable(NNModel.NODE_B_OUT)
        return w_o, b_o

    def build(self):
        '''
        Generates models for controller unit and memory, 
         which will be passed from current step to the next one. 
         Moreover, it creates placeholders for feeding neural network.
        :return:
        '''
        self.__controller.build()
        self.__memory.build()
        self._create_output_nodes()
        print()
        print()

    def _step(self, acc, x_t):
        '''
        :param acc: Accumulator for [c_tm1, h_tm1, m_tm1, r_tm1, wr_tm1, wu_tm1]
        :param x_t: Current input at time t
        :return: None
        '''

        c_tm1, h_tm1, m_tm1, r_tm1, wr_tm1, wu_tm1 = acc
        c_t, h_t, k_t, a_t, s_t = self.__controller.inference(x_t, c_tm1, h_tm1, r_tm1)
        m_t, r_t, wr_t, wu_t = self.__memory.update_mem(m_tm1, wr_tm1, wu_tm1,
                                                        k_t, a_t, s_t)
        # Input to the next step
        return c_t, h_t, m_t, r_t, wr_t, wu_t

    def _offset_input(self, labels):
        '''
         As discussed in Section-2, previous target (y_tp1) is presented as input
         along with the input (x_t) in a temporally offset manner. If you don't want 
         to add this offset the input; then override this method and just return 
         self.__input_ph.
        :param labels: Output 
        :return: 
        '''

        if not self.__is_for_regress:
            labels = tf.one_hot(tf.reshape(labels, [-1]), depth=self.__num_classes)
            labels = tf.reshape(labels, (self.__batch_size,
                                         self.__seq_len,
                                         self.__num_classes))
        # At t=0, (x_0, null) should be the input to NN
        null = tf.slice(labels,
                        [0, 0, 0],
                        [self.__batch_size, 1, self.__num_classes])
        null = tf.zeros_like(null)  # BSx1xnum_classes
        # Remove the last target (label) for it has no matching input like (x_tp1, y_t)
        wht_last_tar = tf.slice(labels,
                                [0, 0, 0],
                                [self.__batch_size,
                                 self.__seq_len-1,
                                 self.__num_classes])
        offset_target = tf.concat([null, wht_last_tar], axis=1)
        # Now concatenate input and offset_target along the 3rd dimension
        act_input_seq = tf.concat([super(NNModel, self).input_ph,
                                   offset_target],
                                  axis=2)  # BSxSLx(input_size+num_classes)
        return act_input_seq

    def inference(self):

        target_ph = super(NNModel, self).target_ph
        out_shape = (self.__batch_size*self.__seq_len,
                     self.__num_classes)
        labels = target_ph if self.__is_for_regress else tf.to_int64(target_ph)
        act_input_seq = self._offset_input(labels)

        print('****************************************************************')
        # Unroll the network for entire sequence of input
        print('Operations for unrolling the network...')
        nn_output = tf.scan(self._step,
                            elems=tf.transpose(act_input_seq, perm=[1, 0, 2]),  # SLxBSx(input_size+num_classes)
                            initializer=(self.__controller.c_0,
                                         self.__controller.h_0,
                                         self.__memory.m_0,
                                         self.__memory.r_0,
                                         self.__memory.wr_0,
                                         self.__memory.wu_0),
                            name="scanning_MLTF_NN",
                            parallel_iterations=1)
        print('****************************************************************')

        # As for computing output, only h_t & r_t suffice:
        # h_t = SLxBSxCS / r_t = SLxBSx(NR.MIL)
        req_output = tf.transpose(tf.concat([nn_output[1], nn_output[3]], axis=2),
                                  perm=[1, 0, 2])  # BSxSLx(CS+NR.MIL)
        # Calculate the predictions:
        [w_o, b_o] = NNModel._get_node_out()
        preact = tf.matmul(tf.reshape(req_output, (self.__batch_size*self.__seq_len, -1)), w_o)
        preact = tf.add(tf.reshape(preact,
                                   (self.__batch_size,
                                    self.__seq_len,
                                    self.__num_classes)),
                        b_o)  # BSxSLx(num_classes)
        preds = tf.reshape(preact, out_shape)  # (BS.SL)x(num_classes)

        self.__labels = tf.reshape(labels, out_shape) if self.__is_for_regress else tf.reshape(labels, [-1])
        self.__preds = preds  # NEVER EVER APPLY SOFT-MAX FOR IT WILL BE APPLIED WHEN LOSS FUNCTION IS DEFINED
        return self

    def loss_func(self):

        if self.__loss:
            return self
        # Create loss function using predictions (logits) and actual output (labels)
        if not self.__is_for_regress:
            labels = tf_utils.create_one_hot_var([self.__labels.get_shape().as_list()[0]],
                                                 self.__num_classes,
                                                 on_idx=self.__labels)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels,
                logits=self.__preds)

            self.__loss = tf.reduce_mean(cross_entropy, name="loss")
        else:
            self.__loss = loss_funcs.huber_m_loss(self.__labels,
                                                  self.__preds,
                                                  self.__percentile,
                                                  name='loss')
        return self

    def train_func(self):

        if self.__train_op:
            return self
        # Create an AdamOptimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.__learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # A function returns the cost against current loss (a.k.a. 'score' in original implementation)
        # and updates parameters in the model.
        self.__train_op = optimizer.minimize(self.__loss,
                                             global_step=global_step,
                                             name="optimizer")
        return self

    @property
    def memory(self):
        return self.__memory

    @property
    def controller(self):
        return self.__controller

    @property
    def labels(self):
        return self.__labels

    @property
    def preds(self):
        return self.__preds

    @property
    def loss(self):
        return self.__loss

    @property
    def train_op(self):
        return self.__train_op
