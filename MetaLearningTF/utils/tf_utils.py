'''
    Copyright 2017-2022 Department of Electrical and Computer Engineering
    University of Houston, TX/USA
    
    This file is part of UHVision Libraries.
    
    UH Vision libraries are free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    UH Vison Libraries are distributed in the hope that it will be useful,
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


def create_one_hot_var(shape, depth, dtype=tf.float32):
    '''
    It creates a one_hot tensor variable. It initializes the first (0th) one_hot bits
     along the the depth dimension. Its dimensionality is equal to (shape,)+depth
    :param shape: (batch_size)x[features]
    :param depth: How many one_hot bits the tf variable is going to have
    :param dtype: Data type of the one_hot tensor variable
    :return: Reference to the newly created one_hot tensor variable
    '''
    on_idx = np.zeros(shape, dtype=np.int32)
    one_hot = tf.one_hot(on_idx, depth, dtype=dtype)
    return one_hot


def erase_els(ref,
              eraser,
              input_,
              indices):
    '''

    :param ref:
    :param eraser:
    :param input_:
    :param indices:
    :return:
    '''

    actual_indices = tf.stack(indices, axis=1)
    temp_ref = tf.Variable(ref.initialized_value())
    updated_ten = tf.scatter_nd_update(temp_ref, actual_indices, eraser)
    updated_ten = tf.multiply(input_, updated_ten)
    return updated_ten


def inc_tensor_els(ref,
                   addition,
                   input_,
                   indices):
    '''

    :param ref:
    :param addition:
    :param input_:
    :param indices:
    :return:
    '''

    actual_indices = tf.stack(indices, axis=1)
    temp_ref = tf.Variable(ref.initialized_value())
    inc_ten = tf.scatter_nd_add(temp_ref, actual_indices, addition)
    updated_ten = tf.add(input_, inc_ten)
    return updated_ten


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


def slice_with_range(input_,
                     dim_range):
    '''

    :param input_:
    :param dim_range:
    :return:
    '''
    shape = dim_range.shape
    begin = []
    size = []
    for d in range(shape[0]):
        begin.append(dim_range[d, 0])
        size.append(dim_range[d, 1]-dim_range[d, 0])
    begin = tf.constant(begin, dtype=tf.int32)
    size = tf.constant(size, dtype=tf.int32)

    return tf.slice(input_, begin, size)
