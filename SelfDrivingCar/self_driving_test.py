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

from data.real_time_data_feeder import RealTimeDataFeeder
from model.modified_alex_net import ModifiedAlexNet
from utils.parallel_net_runner import ParallelNetRunner

IM_W = 400
IM_H = 300
IM_D = 3


def _step(runner):

    print('Deneme')


def test():

    graph = tf.Graph()
    with graph.as_default():
        # All operations should be built into the same graph
        # Create data feeder
        # Data feeder has operations pushed into the graph
        input_ph = tf.placeholder(tf.float32,
                                  self.__input_shape, name="input_ph")
        data_feeder = RealTimeDataFeeder(FLAGS.tf_record_file_name,
                                         FLAGS.num_threads,
                                         FLAGS.num_epochs,
                                         prepare_fields(),
                                         [IM_H, IM_W, IM_D])
        # Specify session configuration options
        sess_config = tf.ConfigProto()
        sess_config.log_device_placement = FLAGS.log_device_placement
        # Network class' arguments (ModifiedAlexNet)
        net_kwargs = {
            'batch_size': 1,
            'num_channels': IM_D,
            'frame_size': [IM_W, IM_H],
            'num_classes': 3,  # Steering Angle, Throttle, and Brake
        }
        # Create Parallel Runner
        # If not specified, Parallel Runner creates
        # a graph and a session object when its 'build' method is called.
        # Use them if you need to. You may have access to these objects
        # through its properties list. Otherwise; specify graph and
        # session objects using its setter functions right after you
        # create an instance of it.
        runner = ParallelNetRunner(
            ModifiedAlexNet,
            net_kwargs,
            data_feeder,
            is_for_training=False,
            is_using_tflearn=True
        )
        runner.config_proto = sess_config  # Specify custom session config options
        runner.graph = graph
        # Build the network.
        # Right after this call, graph and session objects are created
        runner.build()
        # Start execution
        runner.run(_step, None)