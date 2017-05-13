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

from utils import constants as consts
from data.tf_record_feeder import TFRecordFeeder


class TFSeqRecFeeder(TFRecordFeeder):

    __seq_len = None

    __op_incr = None
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
            self.__op_incr = tf.assign_add(var_count, 1)

    def _create_features(self):

        # First record has 'seq_len' # of label tuples; whereas
        # subsequent features has single tuple.
        features = {
            consts.IMAGE_RAW: tf.FixedLenFeature([], self.dtypes[0])
        }
        for name in self.label_fields:
            features[name] = tf.VarLenFeature(self.dtypes[1])
        return features

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
        sess.run([self.__op_incr,
                  self.__op_save_prev_frames,
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
            capacity=min_queue_examples + 2 * batch_size
        )
        return images, labels

    def _prepare_cur_input(self, features):

        var_count = TFSeqRecFeeder._get_var_count()
        # **********************************************************
        # **************************Frames**************************
        # **********************************************************
        prev_frames = TFSeqRecFeeder._get_prev_frames()
        image = tf.decode_raw(features[consts.IMAGE_RAW], tf.uint8)

        def first_fr():
            frame_seq = tf.reshape(image,
                                   shape=[self.__seq_len]+self.image_size)
            return frame_seq

        def next_fr_seq():
            cur_frame = tf.expand_dims(tf.reshape(image, self.image_size),
                                       axis=0)
            removed = prev_frames[1:self.__seq_len, ...]
            appended = tf.concat([removed, cur_frame], axis=0)
            return appended

        cur_frames = tf.cond(tf.less(var_count, 1),
                             first_fr,
                             next_fr_seq)

        # **********************************************************
        # **************************Labels**************************
        # **********************************************************
        prev_labels = TFSeqRecFeeder._get_prev_labels(self.dtypes[1])
        labels = []
        for name in self.label_fields:
            if name != consts.IMAGE_RAW:
                labels.append(tf.sparse_tensor_to_dense(features[name]))

        def first_lb():

            nl = []
            for l in labels:
                nl.append(tf.reshape(l, shape=[self.__seq_len, 1], name='first_one'))
            lb_seq = tf.concat(nl, axis=1)
            return lb_seq

        def next_lb_seq():

            nt = []
            for l in labels:
                nt.append(tf.reshape(l, shape=[1, 1], name='second_one'))
            nt = tf.concat(nt, axis=1)
            removed = prev_labels[1:self.__seq_len, ...]
            appended = tf.concat([removed, nt], axis=0)
            return appended

        cur_labels = tf.cond(tf.less(var_count, 1),
                             first_lb,
                             next_lb_seq)
        cur_labels = tf.cast(cur_labels, self.dtypes[1])

        return cur_frames, cur_labels, self.__op_incr

    def _read_and_decode(self):

        _, serialized_ex = self.reader.read(self.file_name_queue)
        features = tf.parse_single_example(serialized_ex,
                                           features=self.features)
        # Prepare current input
        cur_frames, cur_labels, labels = self._prepare_cur_input(features)

        # Define an operation which saves the previous sequence
        # into the storage variables. By this way, they are accessible by graph operations
        # in the subsequent call to session.run(...)
        prev_frames = TFSeqRecFeeder._get_prev_frames()
        self.__op_save_prev_frames = tf.assign(prev_frames,
                                               cur_frames,
                                               name='op_save_prev_frames')
        prev_labels = TFSeqRecFeeder._get_prev_labels(self.dtypes[1])
        self.__op_save_prev_labels = tf.assign(prev_labels,
                                               cur_labels,
                                               name='op_save_prev_labels')

        # Convolutional layers do not accept data type of uint8
        cur_frames = tf.subtract(
            tf.multiply(tf.cast(cur_frames, tf.float32),
                        tf.constant(1.0 / 255.0)),
            tf.constant(0.5)
        )

        return cur_frames, cur_labels, labels

    def inputs(self,
               batch_size,
               num_ex_per_epoch,
               min_frac_ex_in_queue,
               is_shuffle):
        # Ensure that the shuffling has good mixing properties
        min_queue_examples = int(num_ex_per_epoch*min_frac_ex_in_queue)
        # Read examples from the file
        input_, label, labels2 = self._read_and_decode()
        return input_, label, labels2
