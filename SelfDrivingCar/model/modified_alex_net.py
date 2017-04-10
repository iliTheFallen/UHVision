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
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.optimizers import Momentum

from utils import loss_funcs as loss_func


class ModifiedAlexNet(object):

    # Parameters
    __scope = None
    __batch_size = None
    __num_channels = None
    __frame_size = None
    __input_ph = None
    __target_ph = None
    # -Steering Angle, -Velocity, -Throttle, -Brake
    __num_classes = None
    __learning_rate = None
    __momentum = None
    __percentile = None

    __input_shape = None
    __output_shape = None

    # Network state
    __network = None  # Graph for AlexNet
    __train_op = None  # Operation used to optimize loss function
    __loss = None  # Loss function to be optimized, which is based on predictions
    __total_loss = None  # Loss function after adding regularization losses

    def __init__(self,
                 scope,
                 images=None,
                 labels=None,
                 batch_size=1,
                 num_channels=3,
                 frame_size=(800, 600),
                 num_classes=3,
                 percentile=0.5,
                 learning_rate=0.001,
                 momentum=0.9):

        self.__scope = scope
        self.__batch_size = batch_size
        self.__num_channels = num_channels
        self.__frame_size = frame_size
        self.__num_classes = num_classes
        self.__learning_rate = learning_rate
        self.__momentum = momentum
        self.__percentile = percentile

        self.__input_shape = [self.__batch_size,
                              self.__frame_size[0],
                              self.__frame_size[1],
                              self.__num_channels]
        self.__output_shape = [self.__batch_size, self.__num_classes]

        if not images and not labels:
            self.__input_ph = images
            self.__target_ph = labels
        else:
            self.__input_ph, self.__target_ph = self._create_placeholders()

    def _create_placeholders(self):

        input_ph = tf.placeholder(tf.float32,
                                  self.__input_shape, name="input_ph")
        target_ph = tf.placeholder(tf.float32,
                                   self.__output_shape,
                                   name="output_ph")

        return input_ph, target_ph

    def prepare_dict(self, input_, target):

        feed_dict = {
            self.__input_ph: input_,
            self.__target_ph: target
        }
        return feed_dict

    def inference(self, is_only_features):

        if self.__network:
            return

        # input_shape does not have the dimension for batch size.
        # It is specified in training phase. That is why we have
        # None for the 1st dimension
        # input_shape.insert(0, None)
        # Building network...
        network = input_data(shape=self.__input_shape,
                             placeholder=self.__input_ph)
        network = conv_2d(network,
                          96, 11, strides=4,
                          activation="relu",
                          scope="conv1st",
                          weight_decay=0.0,
                          regularizer='L2')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network,
                          256, 5,
                          activation="relu",
                          scope="conv2nd",
                          weight_decay=0.0,
                          regularizer='L2')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network,
                          384, 3,
                          activation="relu",
                          scope="conv3rd",
                          weight_decay=0.0,
                          regularizer='L2')
        network = conv_2d(network,
                          384, 3,
                          activation="relu",
                          scope="conv4th",
                          weight_decay=0.0,
                          regularizer='L2')
        network = conv_2d(network,
                          256, 3,
                          activation="relu",
                          scope="conv5th",
                          weight_decay=0.0,
                          regularizer='L2')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = fully_connected(network,
                                  4096,
                                  activation="tanh",
                                  scope="fully1st",
                                  weight_decay=0.004,
                                  regularizer='L2')
        network = dropout(network, 0.5)
        network = fully_connected(network,
                                  4096,
                                  activation="tanh",
                                  scope="fully2nd",
                                  weight_decay=0.004,
                                  regularizer='L2')
        network = dropout(network, 0.5)

        if not is_only_features:
            network = fully_connected(network,
                                      self.__num_classes,
                                      activation="linear",
                                      scope="fully3rd",
                                      weight_decay=0.0,
                                      regularizer='L2')

        self.__network = network
        return self

    def loss(self):

        if not self.__loss:
            self.__loss = loss_func.huber_m_loss(self.__target_ph,
                                                 self.__network,
                                                 self.__percentile,
                                                 name='actual_loss')
        return self

    def total_loss(self):

        if not self.__total_loss:
            # Get regularization losses that will be added to the actual loss
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.__scope)
            # Calculate total loss
            self.__total_loss = tf.add(self.__loss,
                                       tf.add_n(reg_losses, 'reg_losses'),
                                       name='total_loss')
        return self

    def train(self):

        if not self.__train_op:
            # decayed_learning_rate = learning_rate *  decay_rate ^ (global_step / decay_steps)
            momentum = Momentum(learning_rate=self.__learning_rate,
                                momentum=self.__momentum)
            self.__train_op = momentum.get_tensor()
        return self

    def get_all(self):

        return self.__loss, self.__total_loss, self.__train_op

