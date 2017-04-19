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
    Date:     4/19/17
    File:     base_model
    Comments: Base class for higher level network constructs
    **********************************************************************************
'''

import tensorflow as tf

import abc

from metalearning_tf.utils import py_utils as pu


class BaseModel(object):

    __input_size = None
    __target_size = None
    __input_type = None
    __target_type = None
    __input_ph = None
    __target_ph = None

    def __init__(self,
                 input_size,
                 target_size,
                 input_type=tf.float32,
                 target_type=tf.float32,
                 inputs=None,
                 targets=None):

        self.__input_size = input_size
        self.__target_size = target_size
        self.__input_type = input_type
        self.__target_type = target_type
        self.__input_ph = inputs
        self.__target_ph = targets

        self._create_placeholders()

    def _create_placeholders(self):

        with tf.name_scope(self.ph_scope):
            if pu.is_empty(self.__input_ph):
                self.__input_ph = tf.placeholder(self.__input_type,
                                                 self.__input_size,
                                                 "input_ph")
            if pu.is_empty(self.__target_ph):
                self.__target_ph = tf.placeholder(self.__target_type,
                                                  self.__target_size,
                                                  "target_ph")

    def prepare_dict(self, input_, target):

        feed_dict = {}
        if not pu.is_empty(input_):
            feed_dict[self.__input_ph] = input_
        if not pu.is_empty(target):
            feed_dict[self.__target_ph] = target
        return feed_dict

    @abc.abstractmethod
    def inference(self):
        raise NotImplementedError("Implement this method in child classes!")

    @abc.abstractmethod
    def loss_func(self):
        raise NotImplementedError("Implement this method in child classes!")

    @abc.abstractmethod
    def train_func(self):
        raise NotImplementedError("Implement this method in child classes!")

    @property
    def ph_scope(self):
        return "placeholders"

    @property
    def input_size(self):
        return self.__input_size

    @property
    def target_size(self):
        return self.__target_size

    @property
    def input_type(self):
        return self.__input_type

    @property
    def target_type(self):
        return self.__target_type

    @property
    def input_ph(self):
        return self.__input_ph

    @property
    def target_ph(self):
        return self.__target_ph
