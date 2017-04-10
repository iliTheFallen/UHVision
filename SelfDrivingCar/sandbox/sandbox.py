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
    File:     sandbox
    Comments: 
    **********************************************************************************
'''

import tensorflow as tf

from data.convert_to_tf_record import ConvertToTFRecord
from data.gtav_data_feeder import GTAVDataFeeder
from utils import loss_funcs as loss
from utils import constants as consts


def test_huber_m_cost():

    targets = [[4, 3],
               [2, 4],
               [5, 5],
               [10, 6]]
    targets = tf.constant(targets, dtype=tf.float32, name="targets")
    labels = [[20, 8],
              [1, 8],
              [5, 8],
              [13, 8]]
    labels = tf.constant(labels, dtype=tf.float32, name="labels")
    loss_tensor = loss.huber_m_loss(labels, targets, 0.5)

    init = tf.global_variables_initializer()
    with tf.Session() as sess, tf.device("/gpu:0"):
        sess.run(init)
        res = sess.run(loss_tensor)
        print('Loss function for 2 features: \n', res)


def convert_to_tf_record():

    reader = GTAVDataFeeder(drive_folder=
                            '/home/cougarnet.uh.edu/igurcan/Documents'
                            '/phdStudies/UHVision/SelfDrivingCar/DriveXbox1')
    types = [  # Order matters
        tf.float32,
        tf.float32,
        tf.float32
    ]
    names = [  # Order matters
        consts.STEERING_ANGLE,
        consts.THROTTLE,
        consts.BRAKE
    ]
    converter = ConvertToTFRecord(reader,
                                  '/home/cougarnet.uh.edu/igurcan/Documents'
                                  '/phdStudies/UHVision/SelfDrivingCar/DriveXbox1',
                                  'gtav_training',
                                  zip(names, types))
    converter.convert(1024)

if __name__ == "__main__":
    # test_huber_m_cost()
    convert_to_tf_record()
