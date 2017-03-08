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
    Date:     2/7/17
    File:     summary_writers
    Comments: This module has methods for attaching tensorflow summary writers in
      order to track variables, activation of units, preactivation of units, and loss
      functions.
    **********************************************************************************
'''

import tensorflow as tf


def attach_to_variable(var, prop_list=('mean', 'stddev', 'min', 'max', 'hist'), scope='summaries'):
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
            tf.summary.scalar("mean", mean)
        if 'stddev' in prop_list:
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
            tf.summary.scalar("stddev", stddev)
        if 'min' in prop_list:
            tf.summary.scalar("min", tf.reduce_min(var))
        if 'max' in prop_list:
            tf.summary.scalar("max", tf.reduce_min(var))
        if 'hist' in prop_list:
            tf.summary.histogram("hist", var)
