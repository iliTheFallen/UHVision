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
    Date:     4/8/17
    File:     mann_cnn_model
    Comments: Represents interconnection of two networks : Convolutional Neural Network
     (CNN) and Memory Augmented Neural Network (MANN). CNN acts like a feature 
     extractor; while MANN is the consumer of these extracted features. They are jointly 
     trained.
    **********************************************************************************
'''

import tensorflow as tf

from model.modified_alex_net import ModifiedAlexNet
from metalearning_tf.model.base_model import BaseModel
from metalearning_tf.model.nn_model import NNModel
from metalearning_tf.utils import py_utils as pu


class MannCnnModel(BaseModel):

    # Configuration options
    __batch_size = None
    __seq_len = None
    __num_channels = None
    __frame_size = None
    __num_classes = None
    __percentile = None
    __learning_rate = None
    __momentum = None

    # Used internally by the class
    __mann = None
    __cnn = None
    __whole_network = None

    def __init__(self,
                 inputs=None,
                 labels=None,
                 batch_size=1,
                 seq_len=80,
                 num_channels=3,
                 frame_size=(300, 400),
                 num_classes=3,
                 percentile=0.5,
                 learning_rate=0.001,
                 momentum=0.9):

        self.__batch_size = batch_size
        self.__seq_len = seq_len
        self.__num_channels = num_channels
        self.__frame_size = frame_size
        self.__num_classes = num_classes
        self.__percentile = percentile
        self.__learning_rate = learning_rate
        self.__momentum = momentum

        input_size = (batch_size,  # Batch size
                      self.__frame_size[0],  # Frame Height
                      self.__frame_size[1],  # Frame Width
                      self.__num_channels)
        target_size = (batch_size,
                       self.__seq_len,
                       self.__num_classes)
        super(MannCnnModel, self).__init__(input_size,
                                           target_size,
                                           inputs=inputs,
                                           targets=labels)

    def inference(self):

        if not pu.is_empty(self.__whole_network):
            return self

        input_to_mann = []
        self.__cnn = []
        input_ = super(MannCnnModel, self).input_ph
        with tf.variable_scope(tf.get_variable_scope()) as var_scope:
            for i in range(self.__seq_len):
                self.__cnn.append(ModifiedAlexNet(
                    input_[:, i, ...],
                    batch_size=self.__batch_size,  # Will feed one sample at a time
                    num_channels=self.__num_channels,
                    frame_size=self.__frame_size,
                    num_classes=self.__num_classes,
                    percentile=self.__percentile,
                    learning_rate=self.__learning_rate,
                    momentum=self.__momentum,
                    is_only_features=True))
                input_to_mann.append(tf.expand_dims(self.__cnn[i].inference().network, axis=1))
                var_scope.reuse_variables()
        input_to_mann = tf.concat(input_to_mann, axis=1)
        input_to_mann = tf.reshape(input_to_mann, shape=[self.__batch_size, self.__seq_len, -1])
        self.__mann = NNModel(inputs=input_to_mann,
                              targets=super(MannCnnModel, self).target_ph,
                              is_for_regress=True,
                              batch_size=self.__batch_size,
                              seq_len=self.__seq_len,
                              input_size=input_to_mann.get_shape().as_list()[2],
                              num_classes=self.__num_classes)
        self.__mann.build()  # Create "graph variables" for controller, memory, and the hosting modules
        self.__mann.inference()  # Create "graph operations" pre-defined for controller, memory, and hosting modules
        self.__whole_network = self.__mann.preds
        return self

    def loss_func(self):

        return self.__mann.loss_func()

    def train_func(self):

        return self.__mann.train_func()

    @property
    def loss(self):
        return self.__mann.loss

    @property
    def train_op(self):
        return self.__mann.train_op

    @property
    def network(self):
        return self.__whole_network
