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
    Date:     3/7/17
    File:     modified_alex_net
    Comments: 
    **********************************************************************************
'''

import tensorflow as tf
from tflearn.layers.core import dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.optimizers import Momentum

from metalearning_tf.model.base_model import BaseModel
from metalearning_tf.utils import loss_funcs as loss_func


class ModifiedAlexNet(BaseModel):

    # Parameters
    __num_classes = None  # -Steering Angle, -Velocity, -Throttle, -Brake
    __learning_rate = None
    __momentum = None
    __percentile = None
    __is_only_features = None

    # Fields used by the class internally
    __layer_names = None

    # Class properties
    __network = None  # Graph for AlexNet
    __train_op = None  # Operation used to optimize loss function
    __loss = None  # Loss function to be optimized, which is based on predictions
    __total_loss = None  # Loss function after adding regularization losses

    def __init__(self,
                 images=None,
                 labels=None,
                 batch_size=2,
                 num_channels=3,
                 frame_size=(300, 400),
                 num_classes=3,
                 percentile=0.5,
                 learning_rate=0.001,
                 momentum=0.9,
                 is_only_features=False):

        self.__num_classes = num_classes
        self.__learning_rate = learning_rate
        self.__momentum = momentum
        self.__percentile = percentile
        self.__is_only_features = is_only_features

        self.__layer_names = [
            'conv1st',
            'conv2nd',
            'conv3rd',
            'conv4th',
            'conv5th',
            'fully1st',
            'fully2nd',
            'fully3rd'
        ]

        input_shape = (batch_size,
                       frame_size[0],
                       frame_size[1],
                       num_channels)
        output_shape = (batch_size, self.__num_classes)

        super(ModifiedAlexNet, self).__init__(input_shape,
                                              output_shape,
                                              inputs=images,
                                              targets=labels)

    def inference(self):

        if self.__network:
            return self

        # input_shape does not have the dimension for batch size.
        # It is specified in training phase. That is why we have
        # None for the 1st dimension
        # input_shape.insert(0, None)
        # Building network...

        # Don't use input layer for it adds extra
        # operations which cause trouble for checkpoint saver.
        network = conv_2d(super(ModifiedAlexNet, self).input_ph,
                          96, 11, strides=4,
                          activation="relu",
                          scope=self.__layer_names[0])
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network,
                          256, 5,
                          activation="relu",
                          scope=self.__layer_names[1])
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network,
                          384, 3,
                          activation="relu",
                          scope=self.__layer_names[2])
        network = conv_2d(network,
                          384, 3,
                          activation="relu",
                          scope=self.__layer_names[3])
        network = conv_2d(network,
                          256, 3,
                          activation="relu",
                          scope=self.__layer_names[4])
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = fully_connected(network,
                                  4096,
                                  activation="tanh",
                                  scope=self.__layer_names[5])
        network = dropout(network, 0.5)
        network = fully_connected(network,
                                  4096,
                                  activation="tanh",
                                  scope=self.__layer_names[6])
        network = dropout(network, 0.5)
        if not self.__is_only_features:
            network = fully_connected(network,
                                      self.__num_classes,
                                      activation="linear",
                                      scope=self.__layer_names[7])

        self.__network = network
        return self

    def loss_func(self):

        if self.__loss:
            return self

        self.__loss = loss_func.huber_m_loss(super(ModifiedAlexNet, self).target_ph,
                                             self.__network,
                                             self.__percentile,
                                             name='actual_loss')
        # Get regularization losses that will be added to the actual loss
        reg_losses = []
        for name in self.__layer_names:
            rl = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, name)
            if rl:
                reg_losses.append(rl)
        # Calculate total loss
        if len(reg_losses) > 0:
            self.__loss = tf.add(self.__loss,
                                 tf.add_n(reg_losses, 'reg_losses'),
                                 name='total_loss')
        return self

    def train_func(self):

        if self.__train_op:
            return self

        # decayed_learning_rate = learning_rate *  decay_rate ^ (global_step / decay_steps)
        momentum = Momentum(learning_rate=self.__learning_rate,
                            momentum=self.__momentum)
        self.__train_op = momentum.get_tensor()
        return self

    @property
    def network(self):
        return self.__network

    @property
    def train_op(self):
        return self.__train_op

    @property
    def loss(self):
        return self.__loss
