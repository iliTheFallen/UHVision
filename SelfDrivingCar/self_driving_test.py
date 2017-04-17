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
    File:     self_driving_test
    Comments: 
    **********************************************************************************
'''

import tensorflow as tf

from scipy.misc import imresize
from queue import LifoQueue
import threading
import numpy as np

from data.gtav_data_reader import GTAVDataReader
from model.modified_alex_net import ModifiedAlexNet
from utils import constants as consts

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("checkpoint_dir", './alexnet_train',
                           """"Directory in where logs and checkpoints are stored""")
tf.app.flags.DEFINE_float("moving_average_decay", 0.9999,
                          """"Decay rate for the past records in exponential moving average""")

IM_W = 400
IM_H = 300
IM_D = 3


def _input():
    ''' Reads a tuple from the data source
    
     It runs in the child thread. Throws tf.errors.OutOfRangeError 
     when the data-stream is exhausted based on a predefined 
     condition.
     
    :return: The next tuple in the data-stream
    '''
    global gtav_reader
    global count
    # print('Reading the next frame...')
    # TODO: Receives the next frame through GTAV network
    # TODO: Do not forget to resize the image to [IM_H, IM_W, IM_D] given above

    count += 1
    if count > 1024:
        raise tf.errors.OutOfRangeError(None, None, "Resource Exhausted")

    frame_buf, _ = gtav_reader.next()
    frame_buf = imresize(frame_buf, size=[IM_H, IM_W, IM_D], interp='bicubic')
    return frame_buf


def _produce(coord, q, alex_net):
    ''' Queues a tuple
    
     Runs in the child thread.
    :param coord: 
    :param queue: 
    :param alex_net: 
    :return: 
    '''

    try:
        while not coord.should_stop():
            feed_dict = alex_net.prepare_dict(
                np.expand_dims(_input(), axis=0).tolist(),
                None
            )
            q.put(feed_dict)
    except KeyboardInterrupt as ex1:
        print('Queueing process is interrupted...')
        coord.request_stop(ex1)
    except tf.errors.OutOfRangeError as ex2:
        print('Frames exhausted...')
        coord.request_stop(ex2)


def _outputs(out):
    ''' Processes the output of neural network in main thread.
     
    :param out: 
    :return: 
    '''
    print((consts.STEERING_ANGLE+": %.4f / " +
          consts.THROTTLE + ": %.4f" +
          consts.BRAKE + ": %.4f") % (out[0], out[1], out[2]))
    # TODO: Send the output to GTAV process through network


def _consume(coord, q, sess, alex_net):
    ''' Consumes the next input data from the given queue.
    
     Runs in main thread. It stops processing when the child thread 
     dies due to one of the exceptions the child thread receives.
    :param coord: 
    :param sess: 
    :param alex_net: 
    :return: 
    '''
    try:
        while not coord.should_stop():
            # print('Dequeuing a frame to process...')
            # While queue is empty wait until it is filled with at least one record.
            # Note that main thread is the one which is suspended in this case.
            feed_dict = q.get()
            out = sess.run(alex_net.network,
                           feed_dict=feed_dict)
            _outputs(out[0])
    except KeyboardInterrupt as ex1:
        print('Dequeuing process is interrupted')
        coord.request_stop(ex1)


def test():

    global gtav_reader
    global count
    gtav_reader = GTAVDataReader(
        drive_folder='/home/ilithefallen/Documents/phdThesis' 
                     '/UHVision/SelfDrivingCar/DriveXbox1')
    count = 0
    with tf.Graph().as_default():
        # Build the model
        alex_net = ModifiedAlexNet(batch_size=1,
                                   num_channels=IM_D,
                                   frame_size=[IM_H, IM_W],
                                   num_classes=3,  # Steering Angle, Throttle, and Brake
                                   is_only_features=False)
        alex_net.inference()
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restore the model
                print('Restoring the model...')
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Finished...')
            else:
                print('No checkpoint path has been found!')
                return
            # Create a Coordinator to deal with exceptions
            coord = tf.train.Coordinator(clean_stop_exception_types=(
                KeyboardInterrupt,
                tf.errors.OutOfRangeError
            ))
            # Create a queue which could be used to process the last entry added to that queue
            # Not exceeds the number of frames per second for we don't want to
            # miss more than that many frames.
            q = LifoQueue(maxsize=1)
            thread = threading.Thread(target=_produce,
                                      args=(coord, q, alex_net))
            coord.register_thread(thread)
            thread.start()  # Start the thread to populate the queue
            _consume(coord, q, sess, alex_net)  # Runs in main thread
            coord.join(stop_grace_period_secs=2)


def main(argv=None):

    # Start testing
    test()


if __name__ == '__main__':
    tf.app.run()