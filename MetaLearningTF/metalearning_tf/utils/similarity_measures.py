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
    Date:     2/6/17
    File:     similarity_measures
    Comments: This module houses all similarity measures used for measuring distance
     between two different data points
    **********************************************************************************
'''

import tensorflow as tf


def cosine_similarity(k_t, m_t):
    # Dot product for similarity
    sim = tf.matmul(k_t, tf.transpose(m_t, perm=[0, 2, 1]))  # BSxNRxMS
    # Outer product for norm multiplication--BSxNRxMS
    norm = tf.sqrt(tf.multiply(
        tf.expand_dims(tf.reduce_sum(tf.multiply(k_t, k_t), 2), 2),   # BSxNRx1
        tf.expand_dims(tf.reduce_sum(tf.multiply(m_t, m_t), 2), 1)))  # BSx1xMS
    # Element-wise division
    return tf.divide(sim, norm)
