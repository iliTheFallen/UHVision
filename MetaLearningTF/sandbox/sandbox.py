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
    Date:     2/20/17
    File:     sandbox
    Comments: Do your test and trials here!
    **********************************************************************************
'''

import tensorflow as tf
import numpy as np
import utils.tf_utils as tf_utils


def test_scatter_nd_add():
    # Testing scatter_nd_add in order to understand what the heck is going on with it!
    test_data = np.asarray([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]],
                           dtype=np.float32)  # Shape=3x3, Rank P=2
    print('Original Array: \n', test_data)
    # What is to be known about indices array is:
    # 1-) Scalars in each innermost dimension correspond to [idx_for(d_0), idx_for(d_1), ..., idx_for(d_N-1)]
    # 2-) Outer dimensions of indices array are implicit indices into updates array
    test_indices = np.asarray([[[0, 2], [0, 1]],
                               [[1, 0], [1, 2]],
                               [[2, 0], [2, 1]]], dtype=np.int32)  # Shape=3x2x2(d_0xd_1xd_2=K), Rank Q=3, K=2
    # Assume that the innermost dimension of "indices" array above is omitted:
    # [[i_00, i_01],
    #  [i_10, i_11],
    #  [i_20, i_22]]
    # As you can see that it is a 3x2 array specifying the indices into updates array below.
    test_updates = np.asarray([[2, 3],
                               [1, -1],
                               [-2, -3]], dtype=np.float32)  # Shape=3x2(d_0xd_{Q-2}), Rank Q-1+P-K = 2

    # By this way, scatter_nd_add can be implemented for the given sample as follow:
    # *********************************************
    # for i in range(num_rows):
    #    for j in range(num_cols):
    #       val_to_add = test_updates[i, j]
    #       idx_into_data = test_indices[i, j, :]
    #       test_data[idx_into_data] += val_to_add
    # *********************************************
    # i.e. one index pair for both "updates" and "indices" arrays
    ref = tf.Variable(test_data.tolist(), dtype=tf.float32)
    indices = tf.constant(test_indices.tolist(), dtype=tf.int32)
    updates = tf.constant(test_updates.tolist(), dtype=tf.float32)

    add = tf.scatter_nd_add(ref, indices, updates)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print('Modified Array: \n', sess.run(add))


def test_update_tensor_els():

    test_data = [[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]]  # Shape=3x3, Rank P=2
    test_data = tf.Variable(test_data, dtype=tf.float32, name="test_data")
    # Testing update
    d_0_idx = tf.constant([0, 0, 1, 1, 2, 2], dtype=tf.int32, name="d0_idx")
    d_1_idx = tf.constant([0, 2, 0, 2, 0, 2], dtype=tf.int32, name="d1_idx")
    idx_list = [d_0_idx, d_1_idx]
    test_updates = [[-1, -2],
                    [-2, -1],
                    [-1, -2]] # Shape=3x2(d_0xd_{Q-2}), Rank Q-1+P-K = 2
    test_updates = tf.Variable(test_updates, dtype=tf.float32, name="test_updates")
    updated = tf_utils.update_tensor_els(test_data,
                                         idx_list,
                                         test_updates)
    # Testing increment
    d_0_idx = tf.constant([0, 2], dtype=tf.int32, name="d0_idx_inc")
    # Unspecified dimension should have the same length as does the original data.
    test_inc = [[-1, -1, -1],
                [-3, -3, -3]]
    test_inc = tf.Variable(test_inc, dtype=tf.float32, name="test_increment")
    increment = tf_utils.inc_tensor_els(test_data,
                                        [d_0_idx],
                                        test_inc)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print('Update-Test Result: \n', sess.run(updated))
        print('Increment-Test Result: \n', sess.run(increment))


if __name__ == "__main__":

    # test_scatter_nd_add()
    test_update_tensor_els()