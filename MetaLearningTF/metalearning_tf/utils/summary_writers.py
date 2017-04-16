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
    Date:     2/7/17
    File:     summary_writers
    Comments: This module has methods for attaching tensorflow summary writers in
      order to track variables, activation of units, preactivation of units, and loss
      functions.
    **********************************************************************************
'''

import tensorflow as tf


import metalearning_tf.utils.tf_utils as tf_utils


def attach_activation_summary(activation, tensor_name='activation'):
    '''
    
     Attaches a summary that provides a histogram of activations.
     Attaches a summary that measures the sparsity of activations.
     
    :param activation: A tensor representing the activation of a layer 
    :param tensor_name: A name for this summary
    :return: None
    '''
    tf.summary.histogram(tensor_name+'/activations', activation)
    tf.summary.scalar(tensor_name+'sparsity', tf.nn.zero_fraction(activation))


def attach_to_variable(var,
                       prop_list=('mean', 'stddev', 'min', 'max', 'hist'),
                       scope='summaries'):
    '''
    
      Attaches summary writer to the given tensorflow variable, so that users
      may view how it makes progress on tensorboard application.

    :param var: Variable to which summary writers will be attached
    :param prop_list: List of properties to be calculated for the given variable
    :param scope: A meaningful name for the variable scope
    :return: void
    '''
    with tf.name_scope(scope):
        mean = tf.reduce_mean(var)
        if 'mean' in prop_list:
            tf.summary.scalar('mean', mean)
        if 'stddev' in prop_list:
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
            tf.summary.scalar('stddev', stddev)
        if 'min' in prop_list:
            tf.summary.scalar('min', tf.reduce_min(var))
        if 'max' in prop_list:
            tf.summary.scalar('max', tf.reduce_min(var))
        if 'hist' in prop_list:
            tf.summary.histogram("hist", var)


def attach_categorical_acc(predictions,
                           labels,
                           scope='summaries',
                           name='accuracy'):
    '''

    :param predictions:
    :param labels:
    :param scope:
    :param name:
    :return:
    '''
    with tf.name_scope(scope):
        classes = tf.cast(labels, tf.int64)
        corr_preds = tf.equal(tf.argmax(predictions, 1), classes)
        accuracy = tf.reduce_mean(tf.cast(corr_preds, tf.float32))
    tf.summary.scalar(name, accuracy)
    return corr_preds, accuracy

def attach_cat_pred_dist(predictions,
                         labels,
                         nb_classes,
                         nb_samples_per_class,
                         batch_size,
                         scope='summaries',
                         name='accuracy'):
    '''
    Attaches predictive probability distribution for categorical outputs
     (i.e. classification)---> p(y_t |x_t, D_(0:t-1); \theta)

    :param predictions:
    :param labels:
    :param nb_classes:
    :param nb_samples_per_class:
    :param batch_size:
    :param scope:
    :param name:
    :return: Each corresponding element in the accuracy array is the answer to the following question
     What is the probability of correctly predicting
     that sample x_t belongs to class y_t
     provided that the class instance
     is currently encountered for the n_th time?
    '''

    acc_0 = tf.zeros([batch_size, nb_samples_per_class], dtype=tf.float32)
    idx_0 = tf.zeros([batch_size, nb_classes], dtype=tf.float32)

    # It is for the recursive operation defined in 'Haskell Language' as: "foldl f z (x:xs) = foldl f (f z x) xs"
    # where xs is a list of items, f is the following step_ function, z is initial condition for the accumulator,
    # and x is the current value of the input to our function f (i.e. step_ given below)
    def step_(a, x):

        # Unpack the variables
        acc, idx = a[:, 0:nb_samples_per_class], a[:, nb_samples_per_class:nb_samples_per_class+nb_classes]
        pre, tar = x[0:batch_size], x[batch_size:2*batch_size]
        # Find indices into acc array
        d_0 = tf.range(0, limit=batch_size, dtype=tf.int64)
        idx_into_idx = tf.stack([d_0, tar], axis=1)
        d_1 = tf.cast(tf.gather_nd(idx, idx_into_idx), tf.int64)
        # Add 1 to correctly predicted classes
        cmp = tf.cast(tf.equal(pre, tar), tf.float32)
        acc_inc = tf_utils.inc_tensor_els(acc, cmp, [d_0, d_1])
        # Increase time instance of each class encountered in this sequence
        idx_inc = tf_utils.inc_tensor_els(idx, tf.ones([batch_size]), [d_0, tar])
        return tf.concat([acc_inc, idx_inc], 1)

    # Compute predictive distribution (i.e. accuracy)
    with tf.name_scope(scope):
        elems = tf.concat([tf.transpose(predictions, perm=[1, 0]),
                           tf.transpose(labels, perm=[1, 0])], 1)
        init = tf.concat([acc_0, idx_0], 1)
        out = tf.foldl(step_,
                       elems,
                       initializer=init,
                       parallel_iterations=1,
                       back_prop=False)
        acc, _ = tf.split(out, [nb_samples_per_class, nb_classes], axis=1)
        acc = tf.reduce_mean(tf.realdiv(acc, tf.constant([nb_classes], dtype=tf.float32)), axis=0)
        # Attach summary writers
        for i in range(nb_samples_per_class):
            acc_i = acc[i]
            tf.summary.scalar(name+"_"+str(i), acc_i)
    return acc

