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
    Date:     4/14/17
    File:     real_time_data_feeder
    Comments: 
    **********************************************************************************
'''

import tensorflow as tf


class RealTimeDataFeeder(object):

    __dec_priority_op = None
    __update_priority_op = None
    __inputs = None
    __capacity = None
    __scope = None

    def __init__(self, inputs, capacity=10, scope=None):

        if inputs is None:
            raise ValueError('inputs tensor cannot be empty!')
        self.__inputs = tf.convert_to_tensor(inputs)  # Will throw exception in the case conversion fails
        self.__capacity = capacity
        self.__scope = scope if scope else "priority_scope"
        with tf.variable_scope(self.__scope):
            self.__priority = tf.get_variable("priority",
                                              shape=[()],
                                              dtype=tf.int64,
                                              initializer=tf.constant_initializer(0, dtype=tf.int64))
        self._define_dec_priority_ops()

    def _define_dec_priority_ops(self):

        # Want to push the operation to the same graph in which 'inputs' in
        with tf.name_scope(self.__scope+'_op', values=[self.__inputs]):
            priority = self.priority
            self.__dec_priority_op = tf.subtract(priority,
                                                 tf.constant(1, dtype=tf.int64, shape=[()]),
                                                 name="dec_priority_op")
            self.__update_priority_op = tf.assign(priority,
                                                  self.__dec_priority_op,
                                                  use_locking=True)

    def after_step(self, sess):
        ''' Executed right after a step of training/testing occurs
        
         Updates priority variable to a higher value
         
        :param sess: 
        :return: 
        '''
        sess.run(self.__update_priority_op)

    def inputs(self):

        # Get types of tensors in inputs
        # Don't include priority variable's data type.
        # PriorityQueue already adds it to 'types' list by itself
        types = []
        if not (isinstance(self.__inputs, list) or isinstance(self.__inputs, tuple)):
            types.append(self.__inputs.dtype)
        else:
            for input_ in self.__inputs:
                types.append(input_.dtype)
        # Create priority queue
        queue = tf.PriorityQueue(self.__capacity, types)
        # We will have one thread to enqueue and dequeue
        enqueue_op = queue.enqueue([self.__dec_priority_op, self.__inputs])
        dequeue_op = queue.dequeue(name="real_time_dequeue_op")
        # Create and add queue runner
        tf.train.queue_runner.add_queue_runner(tf.train.QueueRunner(
            queue,
            [enqueue_op],
            queue_closed_exception_types=(tf.errors.OutOfRangeError,
                                          tf.errors.CancelledError,
                                          KeyboardInterrupt)
        ))
        return dequeue_op, None

    @property
    def priority(self):

        with tf.variable_scope(self.__scope) as var_scope:
            var_scope.reuse_variables()
            return tf.get_variable("priority")