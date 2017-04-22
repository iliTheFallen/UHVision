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
    File:     tf_record_feeder
    Comments: 
    **********************************************************************************
'''

import tensorflow as tf

from utils import constants as consts
from data.base_feeder import BaseFeeder


class TFRecordFeeder(BaseFeeder):

    # Parameters
    __fields = None
    __image_size = None

    # Fields used by the class internally
    __file_name_queue = None
    __reader = None
    __features = None

    def __init__(self,
                 num_threads,
                 tf_record_file_name,
                 num_epochs,
                 fields,
                 image_size):

        self.__fields = fields
        self.__image_size = image_size
        self.__fields = list(fields)  # Avoid exhaustion

        self.__features = {}
        for (name, t) in self.__fields:
            self.__features[name] = tf.FixedLenFeature([], t)
        self.__file_name_queue = tf.train.string_input_producer([tf_record_file_name],
                                                                num_epochs=num_epochs)
        self.__reader = tf.TFRecordReader()
        super(TFRecordFeeder, self).__init__(num_threads)

    def _read_and_decode(self):

        _, serialized_ex = self.__reader.read(self.__file_name_queue)
        features = tf.parse_single_example(serialized_ex,
                                           features=self.__features)
        image = tf.decode_raw(features[consts.IMAGE_RAW], tf.uint8)
        # TODO: Write a comment for why set_shape does not work
        image = tf.reshape(image, self.__image_size)
        # Convolutional layers do not accept data type of uint8
        image = tf.subtract(tf.multiply(tf.cast(image, tf.float32),
                                        tf.constant(1.0/255.0)),
                            tf.constant(0.5))
        labels = []
        for (name, t) in self.__fields:
            if name != consts.IMAGE_RAW:
                labels.append(tf.cast(features[name], t))

        return image, labels
