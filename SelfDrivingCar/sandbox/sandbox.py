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

from PIL import Image

from data.convert_to_tf_record import ConvertToTFRecord
from data.gtav_data_reader import GTAVDataReader
from data.tf_record_feeder import TFRecordFeeder
from metalearning_tf.utils import loss_funcs
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
    loss_tensor = loss_funcs.huber_m_loss(labels, targets, 0.5)

    init = tf.global_variables_initializer()
    with tf.Graph().as_default(), tf.device("/gpu:0"):
        with tf.Session() as sess:
            sess.run(init)
            res = sess.run(loss_tensor)
            print('Loss function for 2 features: \n', res)


def convert_to_tf_record():

    reader = GTAVDataReader(episodeSize=1,
                            max_iter=13056,
                            drive_folder=
                            '/home/ilithefallen/Documents/GTAVDrives')
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
                                  '/home/ilithefallen/Documents/phdThesis'
                                  '/UHVision/SelfDrivingCar/samples',
                                  'gtav_training',
                                  zip(names, types))
    converter.convert(1024, im_size=(300, 400))


def test_parallel_data_feeder():

    def _prepare_fields():
        types = [  # Order matters
            tf.string,
            tf.float32,
            tf.float32,
            tf.float32
        ]
        names = [  # Order matters
            consts.IMAGE_RAW,
            consts.STEERING_ANGLE,
            consts.THROTTLE,
            consts.BRAKE
        ]
        return zip(names, types)
    data_feeder = TFRecordFeeder('/home/ilithefallen/Documents/phdThesis'
                                 '/UHVision/SelfDrivingCar/DriveXbox1'
                                 '/gtav_training.tfrecords',
                                 2,
                                 1,
                                 _prepare_fields(),
                                 [300, 400, 3])
    images, labels = data_feeder.inputs(16, 1024, 0.4, True)
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    with tf.device("/gpu:0"):
        with tf.Session() as sess:
            sess.run(init)
            # Start input enqueue threads for reading input data using 'parallel data feeder' mechanism
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            step = 0
            try:
                while not coord.should_stop():
                    im_out, la_out = sess.run([images, labels])
                    num_im = im_out.shape[0]
                    for i in range(num_im):
                        image = Image.fromarray(im_out[i, :, :, :], 'RGB')
                        image.show()
                    step += 1
            except tf.errors.OutOfRangeError:
                print('Done reading for %d steps.' % (step))
            except KeyboardInterrupt:
                print('Done reading for %d steps.' % (step))
            finally:
                # When done, ask all threads to stop
                coord.request_stop()
            # Wait for threads to finish
            coord.join(threads)


def func1(name, soft_id):

    print('Name %s / SoftId: %d' % (name, soft_id))


def pack_unpack(**kwargs):

    print('Calling the function func1')
    func1(**kwargs)

if __name__ == "__main__":
    # test_huber_m_cost()
    convert_to_tf_record()
    # test_parallel_data_feeder()
    # pack_unpack(name='Ilker GURCAN', soft_id=1456789)
