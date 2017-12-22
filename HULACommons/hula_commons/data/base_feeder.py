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
    Date:     4/21/17
    File:     base_feeder
    Comments: 
    **********************************************************************************
'''

import tensorflow as tf

import abc

from hula_commons.utils import py_utils as pu


class BaseFeeder(object):

    # Parameters
    __num_threads = None

    def __init__(self, num_threads):

        self.__num_threads = num_threads

    @abc.abstractclassmethod
    def _read_and_decode(self):
        '''
        
        :return: 
        '''
        raise NotImplementedError('Please implement this method in child classes!')

    def _generate_batch(self,
                        input_,
                        label,
                        batch_size,
                        min_queue_examples,
                        is_shuffle):

        if is_shuffle:
            images, labels = tf.train.shuffle_batch(
                [input_, label],
                batch_size=batch_size,
                num_threads=self.__num_threads,
                capacity=min_queue_examples+2*batch_size,
                min_after_dequeue=min_queue_examples
            )
        else:
            images, labels = tf.train.batch(
                [input_, label],
                batch_size=batch_size,
                num_threads=self.__num_threads,
                capacity=min_queue_examples+2*batch_size
            )
        # TODO: Display images in the tensorboard visualizer
        # tf.summary.image('images', images)

        return images, labels

    def before_step(self, sess):
        pass

    def after_step(self, sess):
        pass

    def inputs(self,
               batch_size,
               num_ex_per_epoch,
               min_frac_ex_in_queue,
               is_shuffle):
        # Ensure that the shuffling has good mixing properties
        min_queue_examples = int(num_ex_per_epoch*min_frac_ex_in_queue)
        # Read examples from the file
        input_, label = self._read_and_decode()
        return self._generate_batch(input_,
                                    label,
                                    batch_size,
                                    min_queue_examples,
                                    is_shuffle)

    @staticmethod
    def share_input(inputs,
                    labels,
                    batch_size,
                    num_units):

        inputs_list = []
        labels_list = None if pu.is_empty(labels) else []
        num_samples_per_unit = int(batch_size/num_units)
        for i in range(num_units):
            st = i*num_samples_per_unit
            en = st+num_samples_per_unit
            inputs_list.append(inputs[st:en, ...])
            if not pu.is_empty(labels):
                labels_list.append(labels[st:en, ...])

        return inputs_list, labels_list

    @property
    def num_threads(self):
        return self.__num_threads
