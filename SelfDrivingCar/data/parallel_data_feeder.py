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
    File:     parallel_data_feeder
    Comments: 
    **********************************************************************************
'''

import tensorflow as tf

from utils import constants as consts


class ParallelDataFeeder(object):

    __num_threads = None
    __fields = None
    __image_size = None

    __file_name_queue = None
    __features = None

    def __init__(self,
                 tf_record_file_name,
                 num_threads,
                 num_epochs,
                 fields,
                 image_size):

        self.__num_threads = num_threads
        self.__fields = fields
        self.__image_size = image_size

        self.__features = {}
        for (name, t) in fields:
            self.__features[name] = tf.FixedLenFeature([], t)
        # Its name is a bit ambiguous; but it has all records in it.
        # QueueRunner will read records (pair of image & label) until this file
        # is exhausted. And it queues the same file for 'num_epochs' times
        self.__file_name_queue = tf.train.string_input_producer(tf_record_file_name,
                                                                num_epochs=num_epochs)

    def _read_and_decode(self):

        reader = tf.TFRecordReader()
        _, serialized_ex = reader.read(self.__file_name_queue)
        features = tf.parse_single_example(serialized_ex,
                                           features=self.__features)
        image = tf.decode_raw(features[consts.IMAGE_RAW], tf.uint8)
        image.set_shape(self.__image_size)

        labels = []
        for (name, t) in self.__fields:
            if name != consts.IMAGE_RAW:
                labels.append(tf.cast(features[name], t))

        return image, labels

    def _generate_batch(self,
                        image,
                        label,
                        batch_size,
                        min_queue_examples,
                        is_shuffle):

        if is_shuffle:
            images, labels = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=self.__num_threads,
                capacity=min_queue_examples+2*batch_size,
                min_after_dequeue=min_queue_examples
            )
        else:
            images, labels = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=self.__num_threads,
                capacity=min_queue_examples+2*batch_size
            )
        # Display images in the tensorboard visualizer
        tf.summary.image('images', images)

        return images, labels

    def inputs(self,
               batch_size,
               num_ex_per_epoch,
               min_frac_ex_in_queue,
               is_shuffle):

        # Ensure that the shuffling has good mixing properties
        min_queue_examples = int(num_ex_per_epoch*min_frac_ex_in_queue)
        # Read examples from the file
        image, label = self._read_and_decode()
        return self._generate_batch(image,
                                    label,
                                    batch_size,
                                    min_queue_examples,
                                    is_shuffle)


