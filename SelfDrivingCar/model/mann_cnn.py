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
    File:     mann_cnn
    Comments: Represents interconnection of two networks : Memory Augmented Neural 
     Network (MANN) and Convolutional Neural Network (CNN). CNN acts like a feature 
     extractor; while MANN is the consumer of these extracted features. They are jointly 
     trained.
    **********************************************************************************
'''

import tensorflow as tf

from model.modified_alex_net import ModifiedAlexNet
from metalearning_tf.model.base_model import BaseModel
from metalearning_tf.model.nn_model import NNModel
from metalearning_tf.utils import py_utils as pu
from metalearning_tf.utils import loss_funcs


class MannCnn(BaseModel):

    # Configuration options
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
    __loss = None

    def __init__(self,
                 images=None,
                 labels=None,
                 seq_len=80,
                 num_channels=3,
                 frame_size=(300, 400),
                 num_classes=3,
                 percentile=0.5,
                 learning_rate=0.001,
                 momentum=0.9):

        self.__seq_len = seq_len
        self.__num_channels = num_channels
        self.__frame_size = frame_size
        self.__num_classes = num_classes
        self.__percentile = percentile
        self.__learning_rate = learning_rate
        self.__momentum = momentum

        input_size = (1,  # Batch size
                      self.__frame_size[0],  # Frame Height
                      self.__frame_size[1],  # Frame Width
                      self.__num_channels)
        target_size = (self.__seq_len, self.__num_classes)
        super(MannCnn, self).__init__(input_size,
                                      target_size,
                                      inputs=images,
                                      targets=labels)

    def inference(self):

        if not pu.is_empty(self.__cnn):
            return self

        self.__cnn = ModifiedAlexNet(super(MannCnn, self).input_ph,
                                     batch_size=1,  # Will feed one sample at a time
                                     num_channels=self.__num_channels,
                                     frame_size=self.__frame_size,
                                     num_classes=self.__num_classes,
                                     is_only_features=True)
        input_to_mann = self.__cnn.inference().network
        self.__mann = NNModel(inputs=input_to_mann,
                              targets=super(MannCnn, self).target_ph,
                              is_for_regress=True,
                              batch_size=1,
                              seq_len=self.__seq_len,
                              input_size=input_to_mann.get_shape().as_list(),
                              num_classes=self.__num_classes)
        self.__mann.build()

        return self

    def loss_func(self):

        self.__loss = loss_funcs.huber_m_loss(super(MannCnn, self).target_ph,
                                              self.__mann.preds,
                                              self.__percentile,
                                              name='loss')

    def train_func(self):
        pass

    @property
    def loss(self):
        return self.__loss

