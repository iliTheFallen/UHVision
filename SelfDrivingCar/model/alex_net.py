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
    Date:     3/7/17
    File:     alex_net
    Comments: 
    **********************************************************************************
'''

import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.optimizers import Momentum
from tflearn.objectives import categorical_crossentropy


def build_alex_net(input_shape,
                   input_ph,
                   num_classes):

    # input_shape does not have the dimension for batch size.
    # It is specified in training phase. That is why we have
    # None for the 1st dimension
    # input_shape.insert(0, None)
    # Building network...
    network = input_data(shape=input_shape,
                         placeholder=input_ph)
    network = conv_2d(network,
                      96, 11, strides=4,
                      activation="relu",
                      scope="conv1st")
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network,
                      256, 5,
                      activation="relu",
                      scope="conv2nd")
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network,
                      384, 3,
                      activation="relu",
                      scope="conv3rd")
    network = conv_2d(network,
                      384, 3,
                      activation="relu",
                      scope="conv4th")
    network = conv_2d(network,
                      256, 3,
                      activation="relu",
                      scope="conv5th")
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network,
                              4096,
                              activation="tanh",
                              scope="fully1st")
    network = dropout(network, 0.5)
    network = fully_connected(network,
                              4096,
                              activation="tanh",
                              scope="fully2nd")
    network = dropout(network, 0.5)
    # Last layer of the Alexnet will be dropped out for connecting its raw-output to the RNN
    network = fully_connected(network,
                              num_classes,
                              activation="softmax",
                              scope="fully3rd")
    return network


def train(network,
          target_ph,
          learning_rate=0.001,
          momentum=0.9):

    # decayed_learning_rate = learning_rate *  decay_rate ^ (global_step / decay_steps)
    momentum = Momentum(learning_rate=learning_rate,
                        momentum=momentum)
    loss = categorical_crossentropy(network, target_ph)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    train_op = momentum.get_tensor().minimize(loss,
                                              global_step=global_step,
                                              name="minimizer")
    return loss, train_op
