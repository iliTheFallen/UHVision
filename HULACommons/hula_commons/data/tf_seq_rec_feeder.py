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
    Date:     5/6/17
    File:     tf_seq_rec_feeder
    Comments: 
    **********************************************************************************
'''

import tensorflow as tf

import math

from hula_commons.utils import constants as consts
from hula_commons.data.tf_record_feeder import TFRecordFeeder


class TFSeqRecFeeder(TFRecordFeeder):

    __seq_len = None

    __op_inc = None
    __op_save_prev_frames = None
    __op_save_prev_labels = None

    def __init__(self,
                 num_threads,
                 tf_record_file_name,
                 num_epochs,
                 label_fields,
                 dtypes,
                 image_size,
                 seq_len):

        self.__seq_len = seq_len
        super(TFSeqRecFeeder, self).__init__(num_threads,
                                             tf_record_file_name,
                                             num_epochs,
                                             label_fields,
                                             dtypes,
                                             image_size)
        with tf.variable_scope('rec_scope'):
            var_count = tf.get_variable('var_count',
                                        shape=[],
                                        dtype=tf.int64,
                                        trainable=False,
                                        initializer=tf.constant_initializer(value=0, dtype=tf.int64))
            tf.get_variable('prev_frames',
                            shape=[seq_len]+self.image_size,
                            dtype=tf.uint8,
                            trainable=False,
                            initializer=tf.constant_initializer(value=0, dtype=tf.uint8))
            # Create variables for storing labels in the previous sequence
            tf.get_variable('prev_labels',
                            shape=[seq_len, len(self.label_fields)],
                            dtype=dtypes[1],
                            trainable=False,
                            initializer=tf.zeros_initializer(dtypes[1]))
            self.__op_inc = tf.assign_add(var_count, 1)

    @staticmethod
    def _get_var_count():

        with tf.variable_scope('rec_scope') as var_scope:
            var_scope.reuse_variables()
            var_count = tf.get_variable('var_count', dtype=tf.int64)
        return var_count

    @staticmethod
    def _get_prev_frames():

        with tf.variable_scope('rec_scope') as var_scope:
            var_scope.reuse_variables()
            prev_frames = tf.get_variable('prev_frames', dtype=tf.uint8)
        return prev_frames

    @staticmethod
    def _get_prev_labels(t):

        with tf.variable_scope('rec_scope') as var_scope:
            var_scope.reuse_variables()
            prev_labels = tf.get_variable('prev_labels', dtype=t)
        return prev_labels

    def after_step(self, sess):

        # Save the previous 'seq_len' # of frames and labels
        sess.run([self.__op_save_prev_frames,
                  self.__op_save_prev_labels])

    def _generate_batch(self,
                        input_,
                        label,
                        batch_size,
                        min_queue_examples,
                        is_shuffle):

        # For recurrent nets; there is no notion of shuffled batch
        images, labels = tf.train.batch(
            [input_, label],
            batch_size=batch_size,
            num_threads=self.num_threads,
            capacity=min_queue_examples+2*batch_size
        )
        return images, labels

    def _prepare_cur_input(self, features, n):

        # Since features are dense it will look like:
        # [
        #   features {
        #     feature {key: "age" value {int64_list {value: [0]}}}
        #     feature {key: "gender" value {bytes_list {value: ["f"]}}}
        #   },
        #   features {
        #     feature {key: "age" value {int64_list {value: []}}}
        #     feature {key: "gender" value {bytes_list {value: ["f"]}}}
        #   }
        # ]

        # **********************************************************
        # **************************Images**************************
        # **********************************************************
        # This will look like: [0-seq_len], [1-seq_len+1], [2-seq_len+2],...
        # Each slice is composed of SLx(BSxIS) multi-dimensional array.
        img_sub_seqs = [tf.slice(features[consts.IMAGE_RAW],
                                 [i],
                                 [n-self.__seq_len+1]) for i in range(self.__seq_len)]

        # Convolutional layers do not accept data type of uint8
        for i in range(self.__seq_len):
            img_sub_seqs[i] = tf.subtract(
                tf.multiply(tf.cast(img_sub_seqs[i], tf.float32),
                            tf.constant(1.0 / 255.0)),
                tf.constant(0.5)
            )

        # **********************************************************
        # **************************Labels**************************
        # **********************************************************
        lbl_sub_seqs = []
        for name in self.label_fields:
            # Each label category will look like: [0-seq_len], [1-seq_len+1], [2-seq_len+2],...
            # Each slice of any label category is composed of SLx(BSx1) 3-D array.
            lbl_sub_seqs.append([tf.slice(features[name],
                                          [i],
                                          [n-self.__seq_len+1]) for i in range(self.__seq_len)])

        return images, labels

    def _read_and_decode(self):

        # Number of records to read at a time for not to have # of records
        # less than 'seq_len'. Besides it will read records of size that is
        # power of 2.
        n = int(math.ceil(math.log2(self.__seq_len)))
        _, serialized_ex = self.reader.read_up_to(self.file_name_queue, n)
        features = tf.parse_example(serialized_ex,
                                    features=self.features)
        # Prepare current input
        images, labels = self._prepare_cur_input(features, n)

        return images, labels

    # def inputs(self,
    #            batch_size,
    #            num_ex_per_epoch,
    #            min_frac_ex_in_queue,
    #            is_shuffle):
    #     # Ensure that the shuffling has good mixing properties
    #     min_queue_examples = int(num_ex_per_epoch*min_frac_ex_in_queue)
    #     # Read examples from the file
    #     input_, label = self._read_and_decode()
    #     return input_, label, TFSeqRecFeeder._get_var_count()
