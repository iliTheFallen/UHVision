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


def create_one_hot_var(shape, depth, dtype=tf.float32, name=''):
    '''
    It creates a one_hot tensor variable. It initializes the first (0th) one_hot bits
     along the the depth dimension. Its dimensionality is equal to (shape,)+depth
    :param shape: (batch_size)x[features]
    :param depth: How many one_hot bits the tf variable is going to have
    :param dtype: Data type of the one_hot tensor variable
    :param name: Name of the one_hot tensor variable
    :return: Reference to the newly created one_hot tensor variable
    '''
    on_idx = np.zeros(shape, dtype=np.int32)
    one_hot = tf.Variable(tf.one_hot(on_idx, depth, dtype=dtype),
                          name=name)
    return one_hot


def _restructure_indexing(indices_list,
                          updates):
    '''

    :param indices_list:
    :param updates:
    :return:
    '''
    k = len(indices_list)  # Innermost dim of the indices array
    shape_up = updates.get_shape().as_list()
    u = len(shape_up)  # Rank of the update matrix
    # k # of dimensions are required to index "updates" list
    idx_shape = np.split(np.asarray(shape_up), [k, u])[0].tolist()
    # Dimension in which index into "input_" is stored
    idx_shape.append(k)
    actual_indices = tf.stack(indices_list, axis=1)
    actual_indices = tf.reshape(actual_indices, idx_shape)

    return actual_indices


def update_tensor_els(input_,
                      indices_list,
                      updates):
    '''

    :param input_:
    :param indices_list:
    :param updates:
    :return:
    '''
    actual_indices = _restructure_indexing(indices_list, updates)
    updated_ten = tf.scatter_nd_update(input_, actual_indices, updates)

    return updated_ten


def inc_tensor_els(input_,
                   indices_list,
                   addition):
    '''

    :param input_:
    :param indices_list:
    :param addition:
    :return:
    '''
    actual_indices = _restructure_indexing(indices_list, addition)
    updated_ten = tf.scatter_nd_add(input_, actual_indices, addition)

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
    _,sorted_idx = tf.nn.top_k(input_, k=ten_shape[num_dims - 1]).indices
    size = []
    begin = np.zeros(num_dims, dtype=np.int32).tolist()  # Always begins at the 0th idx
    begin = tf.constant(begin)
    # Get all the data along all dimensions except the last one
    for i in range(num_dims-1):
        size.append(ten_shape[i])
    size.append(num_els)
    size = tf.constant(size)
    if is_ascending:
        sorted_idx = tf.reverse(sorted_idx, num_dims-1)

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
