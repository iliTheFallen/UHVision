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

from tflearn import DNN
from tflearn.initializations import normal as normal
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.optimizers import Momentum
from tflearn.layers.estimator import regression

import tensorflow as tf

NUM_EPOCHS=10


def build_alex_net(input_shape,
                   num_classes):

    print('Building AlexNet...')
    # input_shape does not have the dimension for batch size.
    # It is specified in training phase. That is why we have
    # None for the 1st dimension
    # input_shape.insert(0, None)
    # Building network...
    network = input_data(shape=input_shape)
    weight_init = normal(mean=0.0, stddev=0.01)
    network = conv_2d(network,
                      96, 11, strides=4,
                      activation='relu',
                      weights_init=weight_init)
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network,
                      256, 5, activation='relu',
                      weights_init=weight_init,
                      bias_init=tf.constant_initializer(1.0))
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network,
                      384, 3, activation='relu',
                      weights_init=weight_init)
    network = conv_2d(network,
                      384, 3, activation='relu',
                      weights_init=weight_init,
                      bias_init=tf.constant_initializer(1.0))
    network = conv_2d(network,
                      256, 3, activation='relu',
                      weights_init=weight_init,
                      bias_init=tf.constant_initializer(1.0))
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network,
                              4096, activation='tanh',
                              weights_init=weight_init)
    network = dropout(network, 0.5)
    network = fully_connected(network,
                              4096, activation='tanh',
                              weights_init=weight_init)
    network = dropout(network, 0.5)
    network = fully_connected(network,
                              num_classes, activation='softmax',
                              weights_init=weight_init)
    # decayed_learning_rate = learning_rate *  decay_rate ^ (global_step / decay_steps)
    momentum = Momentum(learning_rate=0.01,
                        lr_decay=0.0005,
                        decay_step=100,
                        momentum=0.9)
    network = regression(network,
                         optimizer=momentum,
                         loss='categorical_crossentropy')
    return network


def train(network,
          input_,
          labels,
          batch_size,
          model_name):

    print('Training AlexNet...')
    model = DNN(network,
                checkpoint_path=model_name,
                max_checkpoints=1,
                tensorboard_verbose=2)
    model.fit(input_,
              labels,
              n_epoch=NUM_EPOCHS,
              validation_set=0.1,  # 10% of training data will be used for validation
              shuffle=True,
              show_metric=True,
              batch_size=batch_size,
              snapshot_step=batch_size,  # Save the model at every this many steps.
              snapshot_epoch=False,  # We don't want snapshot at the end of each epoch
              run_id=model_name)
    return model
