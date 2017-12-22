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
    Date:     2/16/17
    File:     tf_utils
    Comments:
    **********************************************************************************
'''

import tensorflow as tf
import numpy as np


def glorot_uniform_init(shape, dtype=tf.float32):
    '''

    :param shape:
    :param dtype:
    :return:
    '''
    shape = np.array(shape)
    if isinstance(shape, int):
        high = np.sqrt(6. / shape)
    else:
        high = np.sqrt(6. / (np.sum(shape[:2]) * np.prod(shape[2:])))
    return tf.random_uniform_initializer(minval=-1*high, maxval=high, dtype=dtype)


def create_one_hot_var(shape, depth, dtype=tf.float32, on_idx=None):
    '''
    It creates a one_hot tensor variable. It initializes the first (0th) one_hot bits
     along the the depth dimension. Its dimensionality is equal to (shape,)+depth
    :param shape: (batch_size)x[features]
    :param depth: How many one_hot bits the tf variable is going to have
    :param dtype: Data type of the one_hot tensor variable
    :param on_idx:
    :return: Reference to the newly created one_hot tensor variable
    '''
    if on_idx is None:
        on_idx = np.zeros(shape, dtype=np.int32)
    one_hot = tf.one_hot(on_idx, depth, dtype=dtype)
    return one_hot


def update_var_els(input_,
                   updates,
                   indices):
    '''

    :param input_:
    :param updates:
    :param indices:
    :return:
    '''

    actual_indices = tf.stack(indices, axis=1)
    old_vals = tf.gather_nd(input_, actual_indices)
    nullifier_updates = tf.subtract(updates, old_vals)
    sparse_updates = tf.scatter_nd(actual_indices,
                                   nullifier_updates,
                                   input_.get_shape().as_list())
    return tf.add(input_, sparse_updates)


def inc_tensor_els(input_,
                   addition,
                   indices):
    '''

    :param input_:
    :param addition:
    :param indices:
    :return:
    '''

    actual_indices = tf.stack(indices, axis=1)
    increased = tf.scatter_nd(actual_indices,
                              addition,
                              input_.get_shape().as_list())
    return tf.add(input_, increased)


def get_sorted_idx(input_,
                   ten_shape,
                   num_els,
                   is_ascending=True):
    '''

    :param input_:
    :param ten_shape:
    :param num_els:
    :param is_ascending:
    :return:
    '''
    # Always sorts in descending order
    num_dims = len(ten_shape)
    _,sorted_idx = tf.nn.top_k(input_, k=ten_shape[num_dims - 1])
    size = []
    begin = np.zeros(num_dims, dtype=np.int32).tolist()  # Always begins at the 0th idx
    begin = tf.constant(begin)
    # Get all the data along all dimensions except the last one
    for i in range(num_dims-1):
        size.append(ten_shape[i])
    size.append(num_els)
    size = tf.constant(size)
    if is_ascending:
        sorted_idx = tf.reverse(sorted_idx, [num_dims-1])

    return tf.slice(sorted_idx, begin, size)

