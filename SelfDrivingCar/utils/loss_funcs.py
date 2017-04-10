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
    Date:     4/7/17
    File:     loss_funcs
    Comments: Missing or Custom Loss functions are defined in this module.
            1-Huber-M Loss Function
    **********************************************************************************
'''

import tensorflow as tf

from metalearning_tf.utils import tf_utils as meta_utils


def huber_m_loss(labels_tensor,
                 preds_tensor,
                 percentile,
                 name='loss'):
    '''
     Calculates the Huber-M loss function for the given 
     parameters. Note that this is for regression purposes; but not 
     for classification.
    :param labels_tensor: Correct predictions. NxF matrix where 
     N is number of targets and F is number of features
    :param preds_tensor: Predicted continuous values by the model.
     It should have the same dimensions as does labels_tensor
    :param percentile: Percentile of residuals in descending order
     on which LAD (Least Absolute Deviation) loss will be applied.
    :param name: Operation name
    :return: Huber-M Loss tensor
    '''
    # Step-1) Find absolute difference between labels & predictions (residuals)
    res = tf.abs(tf.subtract(labels_tensor, preds_tensor))
    [N, F] = res.get_shape().as_list()
    samp_frac = int(round(N*percentile))
    loss = []
    # Step-2) Compute Huber-M Loss for each feature column
    for i in range(F):
        cur_f = res[:, i]
        sorted_idx = meta_utils.get_sorted_idx(cur_f,
                                               [N],
                                               N,
                                               is_ascending=False)
        # Least Absolute Deviation
        lad_idx = sorted_idx[0:samp_frac]
        lad_res = tf.gather(cur_f, lad_idx)
        delta = lad_res[-1]
        lad_res = tf.multiply(delta, tf.subtract(lad_res,
                                                 tf.multiply(0.5, delta)))
        lad_res = tf.reduce_sum(lad_res)
        # Least Square Deviation
        lsd_idx = sorted_idx[samp_frac:]
        lsd_res = tf.gather(cur_f, lsd_idx)
        lsd_res = tf.reduce_sum(tf.multiply(tf.square(lsd_res), 0.5))
        loss.append(tf.realdiv(tf.add(lad_res, lsd_res),
                               tf.constant(N, dtype=tf.float32)))

    # Step-3) Finally merge all these losses into a single loss vector
    loss = tf.stack(loss, 0, name=name)

    return loss

